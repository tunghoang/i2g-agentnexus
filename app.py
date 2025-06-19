import time, signal
import traceback

is_running = True
def mcp_sig_handler(sig, frame):
    global is_running
    is_running = False

def run_mcp_server():
    from servers.mcp_server import MCPServerManager, MCPClient
    from config.settings import Config, load_config
    appconfig = load_config()
    server = MCPServerManager(config=appconfig.mcp, data_config=appconfig.data)
    server.start()
    if server.wait_ready(timeout=60):
        print(f"Server is ready! url={server.url}")
        mcp_client = MCPClient(server.url)
        # Test tool access

    else:
        print("Server not ready")
    signal.signal(signal.SIGINT, mcp_sig_handler)
    global is_running
    while is_running:
        time.sleep(1)

    server.stop()
    print("Server stopped")

def _create_agent_with_fallback(appconfig, mcp_client, tools_response):
    from agents.google_adk_hybrid_agent import create_google_adk_hybrid_agent
    # Create Google ADK HybridAgent using your existing pattern
    agent = create_google_adk_hybrid_agent(
        mcp_url=appconfig.mcp.url,
        config=appconfig.agent
    )

    print("Google ADK HybridAgent created successfully")
    return agent

def create_agent():
    from servers.mcp_server import MCPClient
    from config.settings import Config, load_config

    appconfig = load_config()
    mcp_client = MCPClient(appconfig.mcp.url)

    try:
        tools_response = mcp_client.get_tools()
        if "error" in tools_response:
            raise Exception(f"Tools not immediately available: {tools_response['error']}")

        agent = _create_agent_with_fallback(appconfig, mcp_client, tools_response)

        # Wrap agent to add compatibility methods if needed
        if not hasattr(agent, 'get_stats'):
            raise Exception("agent has no get_stats")
        if not hasattr(agent, 'run'):
            raise Exception("agent has no run")
        print("Agent created successfully")
        return agent
    except:
        traceback.print_exc()

    return None

saved_sig_handler = None
cleaner = None
def agent_sig_handler(sig, frame):
    global cleaner, saved_sig_handler
    if cleaner:
        cleaner.stop()
    if saved_sig_handler:
        saved_sig_handler(sig, frame)
def run_agent_server():
    from qa_server import qa_server_create
    from cleaner import cleaner_create
    import uvicorn
    global cleaner, saved_sig_handler
    saved_sig_handler = signal.signal(signal.SIGINT, agent_sig_handler)
    app_server = qa_server_create(create_agent)
    if app_server is not None:
        cleaner = cleaner_create(app_server)
        cleaner.run()
        uvicorn.run(app_server, port=8990)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--mcp', action="store_true")
    parser.add_argument('--agent', action="store_true")
    args = parser.parse_args()
    if args.mcp:
        run_mcp_server()
    elif args.agent:
        run_agent_server()
    else:
        parser.print_help()
