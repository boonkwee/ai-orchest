"""
Gold Price Agent
A LangGraph agent that fetches live gold prices using a tool.
This introduces: tools, ToolNode, conditional edges, and message-based state.
"""

# GOLD_API_KEY
import os
import requests
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# --- Tool ---
# This is the function the agent can call to get live gold price data.
# # The @tool decorator registers it as a tool that DeepSeek can use.

@tool
def get_gold_price() -> str:
    """Fetch the current gold price in USD per troy ounce, including
    today's change and percentage change from the previous close."""

    api_key = os.environ.get("GOLD_API_KEY")
    if not api_key:
        return "Error: GOLD_API_KEY environment variable is not set."

    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {"x-access-token": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        return (
            f"Gold price: ${data['price']:.2f} per troy ounce. "
            f"Previous close: ${data['prev_close_price']:.2f}. "
            f"Change: ${data['ch']:+.2f} ({data['chp']:+.2f}%). "
            f"Day range: ${data['ch']:+.2f} - ${data['high_price']:.2f}. "
            f"24K oer gram: ${data['price_gram_24k']:.2f}."
        )
    except requests.RequestException as e:
        return f"Error fetching gold price: {e}"
    except (KeyError, TypeError) as e:
        return f"Error parsing gold price data: {e}"

load_dotenv()  # Load environment variables from .env file
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")  # Get the API key from environment variables

# --- LLM ---
# Bind the tools to DeepSeek so it knows they are available.
tools = [get_gold_price]
llm = ChatDeepSeek(model="deepseek-flash", api_key=deepseek_api_key).bind_tools(tools)


# --- Nodes ---
# The agent node: sends the conversation to DeepSeek (with tool access).
def agent(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# The tool node: executes any tool calls that DeepSeek requested.
tool_node = ToolNode(tools)

# --- Conditional edge ---
# After the agent runs, check: did DeepSeek request a tool call?
# If yes - route to the tool node. If no - we're done.
def should_use_tool(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# --- Graph ---
# Build the graph with the agent-tool loop
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tool, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

agent_app = graph.compile()

# --- Run ---
if __name__ == '__main__':
    print("What is the current gold price?")
    result = agent_app.invoke(
        {"messages": [("human", "What is the current gold price? Give me a brief market summary.")]}
    )
    print(result["messages"][-1].content)
