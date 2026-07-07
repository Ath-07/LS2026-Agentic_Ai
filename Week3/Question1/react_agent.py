"""
ReAct Agent with LangChain
===========================
An agent that reasons step-by-step (Thought -> Action -> Observation) and
decides which tool to call in order to answer a user's question.

Tools included (2, as required):
    1. Calculator   - evaluates arithmetic / math expressions
    2. WebSearch     - DuckDuckGoSearchRun (free, no API key needed)

The LLM itself DOES need credentials (it's the "brain" doing the reasoning).
This script is written to work with Anthropic, OpenAI, or Google Gemini
models - just set the matching environment variable and leave only one
LLM block uncommented (Anthropic is uncommented by default).

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."      # if using Claude
    # or
    export OPENAI_API_KEY="sk-..."             # if using GPT models
    # or
    export GOOGLE_API_KEY="AIza..."            # if using Gemini models
    python react_agent.py
"""

import os
import math
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file (if present)

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun


# ---------------------------------------------------------------------------
# 1. Choose the LLM (the "reasoning engine" behind the ReAct loop)
# ---------------------------------------------------------------------------
# --- Option A: Anthropic Claude (default) ---
# from langchain_anthropic import ChatAnthropic

# llm = ChatAnthropic(
#     model="claude-sonnet-4-6",
#     temperature=0,
# )

# --- Option B: OpenAI GPT (swap in instead, if preferred) ---
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Option C: Google Gemini (swap in instead, if preferred) ---
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# # requires: pip install langchain-google-genai==2.1.12
# # requires: export GOOGLE_API_KEY="AIza..."


# ---------------------------------------------------------------------------
# 2. Tool #1 - Calculator
# ---------------------------------------------------------------------------
def calculator(expression: str) -> str:
    """Safely evaluate a math expression and return the result as a string."""
    allowed_names = {
        "sqrt": math.sqrt,
        "pow": pow,
        "abs": abs,
        "round": round,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        # No builtins exposed -> only arithmetic + the whitelisted math funcs above
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description=(
        "Use this for ANY math: arithmetic, percentages, square roots, "
        "exponents, trig, etc. Input must be a single valid Python math "
        "expression, e.g. '234 * 89 + sqrt(144)'. It has no knowledge of "
        "facts, people, or current events -- numbers only."
    ),
)


# ---------------------------------------------------------------------------
# 3. Tool #2 - Web Search (DuckDuckGo, free, no key required)
# ---------------------------------------------------------------------------
_search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="WebSearch",
    func=_search.run,
    description=(
        "Use this to look up current events, facts, people, places, prices, "
        "statistics, or anything that requires up-to-date information from "
        "the internet. Input should be a concise search query string."
    ),
)

tools = [calculator_tool, search_tool]


# ---------------------------------------------------------------------------
# 4. The classic ReAct prompt (hand-written, so no internet fetch from
#    LangChain Hub is required -- keeps this script fully self-contained)
# ---------------------------------------------------------------------------
REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # <-- prints full Thought/Action/Observation trace
    handle_parsing_errors=True,
    max_iterations=6,
)


def ask(query: str) -> str:
    """Run the ReAct agent on a single query string and return the final answer."""
    result = agent_executor.invoke({"input": query})
    return result["output"]


# ---------------------------------------------------------------------------
# 5. Demo -- 3 queries that together force both tools to be used at least once
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_queries = [
        # Forces Calculator only
        "What is 234 multiplied by 89, then add the square root of 144?",
        # Forces WebSearch only
        "Who is the current CEO of Microsoft?",
        # Forces WebSearch THEN Calculator (multi-tool reasoning chain)
        "Search for the current population of Japan, then divide that number by 1000 "
        "to express it in thousands.",
    ]

    for i, q in enumerate(demo_queries, start=1):
        print("\n" + "=" * 80)
        print(f"QUERY {i}: {q}")
        print("=" * 80 + "\n")
        answer = ask(q)
        print(f"\n>>> FINAL ANSWER: {answer}\n")