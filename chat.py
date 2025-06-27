from requests import post, get
import argparse
import dotenv
import os

dotenv.load_dotenv()

AGENT_URL = os.getenv("AGENT_URL") or "http://localhost:8990"

parser = argparse.ArgumentParser()
parser.add_argument("--ask", type=str)
parser.add_argument("--show-agents", action="store_true")
parser.add_argument("--new-agent", action="store_true")
parser.add_argument("--agent-id", type=str)
args = parser.parse_args()

if args.ask:
    response = post(f"{AGENT_URL}/ask", json={"question": args.ask, "agentid": args.agent_id})
    print(response.json())
elif args.show_agents:
    response = get(f"{AGENT_URL}/agents")
    print(response.json())
elif args.new_agent:
    response = get(f"{AGENT_URL}/new")
    print(response.json())
else:
    print("No question provided")