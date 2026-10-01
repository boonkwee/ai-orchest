import os
import requests
import sqlite3
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

@tool
def get_gold_price() -> str:
    """Fetch the current gold price in USD per troy ounce, include
    today's change and percentage change from the pevious close."""

    api_key = os.environ.get("GOLD_API_KEY")
    if not api_key:
        return "Error: GOLD_API_KEY environment variable is not set."

    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {"x-access-token" : api_key}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        return (
            f"Gold price: ${data['price']:.2f} per troy ounce. "
            f"Previous close: ${data['prev_close_price']:.2f}. "
            f"Change: ${data['ch']:+.2f} ({data['chp']:.2f}%)"
            f"Day range: ${data['low_price']:.2f} - ${data['high_price']:.2f}. "
            f"24K per gram: ${data['price_gram_24k']:.2f}."
        )
    except requests.RequestException as e:
        return f"Error fetching gold price: {e}"
    except (KeyError, TypeError) as e:
        return f"Error parsing gold price data: {e}"

# --- LLMs with different system prompts ---
# Each agent gets its own LLM instance with a specialised role.

tools = [get_gold_price]

research_llm = ChatDeepSeek(model='deepseek-flash', api_key=deepseek_api_key).bind_tools(tools=tools)

analysis_llm = ChatDeepSeek(model='deepseek-flash', api_key=deepseek_api_key)

RESEARCH_PROMPT = """You are a Gold Market Research Agent. Your job is to gather
accurate, factual gold price data. When asked about gold prices, ALWAYS use your
get_gold_price tool to fetch live data. Be thorough and include all available
data points: price, change, percentage, day range, and per-gram price.
Present the raw data clearly without adding analysis or opinions."""

ANALYSIS_PROMPT = """You are a Gold Market Analysis Agent. Your job is to interpret
gold price data and provide insightful analysis. You do NOT have access to live
data tools - you work with the data already gathered in the conversation.
Provide observations about:
- Whether the price movement is significant
- What the day's range suggests about volatility
- A brief outlook or context for the price level
Keep your analysis concise and actionable. Think like a financial analyst
writing a morning briefing."""

# --- Nodes ---
def research_agent(state: MessagesState) -> MessagesState:
    """The research agent: gathers data using tools."""
    msg_count = len(state["messages"])
    print(f" - [Research Agent] Gathering data... ({msg_count} messages in memory)")
    messages = [SystemMessage(content=RESEARCH_PROMPT)] + state['messages']
    response = research_llm.invoke(messages)
    return {"messages": [response]}

def analysis_agent(state: MessagesState) -> MessagesState:
    """The analysis agent: interprets data and generates insights."""
    msg_count = len(state["messages"])
    print(f" - [Analysis Agent] Generating insights... ({msg_count} messages in memory)")
    messages = [SystemMessage(content=ANALYSIS_PROMPT)] + state['messages']
    messages.append(HumanMessage(content="Based on the data and conversation above, provide your market analysis."))
    response = analysis_llm.invoke(messages)
    return {"messages": [response]}

# Tool node for the research agent
tool_node = ToolNode(tools=tools)

# --- Keywords for routing ---
DATA_KEYWORDS = ["price", "cost", "worth", "how much", "current",
"today", "rigth now", "latest", "check"]

ANALYSIS_KEYWORDS = ["analyse", "analyze", "tell us", "what does",
"insight", "outlook", "investment", "should i", "recommend",
"opinion", "think about", "suggest", "implication", "mean"]

# --- Router ---
def route_question(state: MessagesState) -> str:
    """Decide which agent handles the question.
    If it nees live data -> research agent.
    If it only needs analysis -> analysis agent."""
    last_message = state["messages"][-1]
    content = last_message.content.lower() if hasattr(last_message, 'content') else ""

    needs_data = any(kw in content for kw in DATA_KEYWORDS)

    if needs_data:
        print(" - [Router] Data needed - Research Agent")
        return "research_agent"
    print(" - [Router] Analysis only - Analysis Agent")
    return "analysis_agent"

def after_research(state: MessagesState) -> str:
    """After the research agent runs, decide next step:
    1. If tool call requested - go to tools
    2. If original question also needs analysis - go to analysis
    3. Otherwise - END (data only)"""
    last_message = state["messages"][-1]

    # If theresearch agent wants to call a tool, do it
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Check if the original question also needs analysis
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            content = msg.content.lower()
            if any(kw in content for kw in ANALYSIS_KEYWORDS):
                print(" - [Handoff] Research complete - Analysis Agent")
                return "analysis_agent"
            break

    print(" - [Complete] Data delivered - no analysis needed")
    return END

# --- Graph ---
graph = StateGraph(MessagesState)

# Add all nodes
graph.add_node("research_agent", research_agent)
graph.add_node("analysis_agent", analysis_agent)
graph.add_node("tools", tool_node)

# Routing from START
graph.add_conditional_edges(
    START, route_question,
    {"research_agent": "research_agent", "analysis_agent": "analysis_agent"}
)

# After research agent: tool, analyst, or done
graph.add_conditional_edges(
    "research_agent", after_research,
    {"tools": "tools", "analysis_agent": "analysis_agent", END: END}
)

# After tools: back to research agent (to see the result)
graph.add_edge("tools", "research_agent")

# Analysis agent always ends the graph
graph.add_edge("analysis_agent", END)


conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
# Compile with memory
checkpointer = SqliteSaver(conn=conn)
app = graph.compile(checkpointer=checkpointer)

# --- Run ---
if __name__=='__main__':
    config = {"configurable": {"thread_id": "multi-agent-session-1"}}
    # Test 1: Data only
    Q1 = "What was the gold price yesterday?"
    print(f"Q1: {Q1}")
    result = app.invoke(
        {"messages": [HumanMessage(content=Q1
        )]},
        config=config
    )
    print(result["messages"][-1].content)
    print()

    # Test 2: Data + analysis - research gathers data, analyst interprets it
    Q2 = "What is the current gold price and what does it tell us?"
    print(f"Q2: {Q2}")
    result = app.invoke(
        {"messages": [HumanMessage(content=Q2
        )]},
        config=config
    )
    print(result["messages"][-1].content)

    Q3 = "Based on what we've discussed, is gold a good investment?"
    print(f"Q3: {Q3}")
    result = app.invoke(
        {"messages": [HumanMessage(
            content=Q3
        )]},
        config=config
    )