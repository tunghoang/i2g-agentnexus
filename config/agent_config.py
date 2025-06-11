"""
Agent Configuration Settings
"""

# Agent implementation selection
AGENT_TYPE = "google_adk_hybrid"  # NEW: Google ADK-powered HybridAgent

# Available agents
AVAILABLE_AGENTS = {
    "google_adk_hybrid": "Google ADK HybridAgent - LangChain replacement",
    "hybrid": "Original HybridAgent (with LangChain issues)",
    "emergency": "Emergency Agent - Basic functionality only"
}

# Google ADK configuration
GOOGLE_ADK_CONFIG = {
    "model": "openai/gpt-4o-mini",
    "temperature": 0.1,
    "max_iterations": 15,
    "verbose": True
}