from typing import TypedDict, Literal

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agent.chains.planning import Planner
from agent.graph.utils.state import State
from agent.chains.simplifier import Simplifier
from config import CONFIG


class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0.3)
# model = 'gpt-4o-2024-08-06'

model = 'gpt-4o-mini-2024-07-18'
llm = ChatOpenAI(model_name=model, temperature=0.2, openai_api_key=CONFIG.openai_api_key)
simplifier_agent = Simplifier(llm)
planner_agent = Planner(llm)

workflow = StateGraph(State, config_schema=GraphConfig)

workflow.set_entry_point("simplifier_agent")
workflow.add_node("simplifier_agent", simplifier_agent)
workflow.add_node("planner_agent", planner_agent)
workflow.add_edge("simplifier_agent", "planner_agent")
workflow.add_edge("planner_agent", END)

graph = workflow.compile()
