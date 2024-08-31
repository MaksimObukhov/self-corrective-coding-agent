from typing import TypedDict, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agent.graph.utils.state import State
from agent.chains.planning import PlanningAgent
from agent.chains.retrieval import SelfRetrieverAgent
from agent.chains.simplifier import SimplifierAgent
from config import CONFIG


class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0.3)
# model = 'gpt-4o-2024-08-06'

model = 'gpt-4o-mini-2024-07-18'
llm = ChatOpenAI(model_name=model, openai_api_key=CONFIG.openai_api_key).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

# Agents
simplifier_agent = SimplifierAgent(llm)
retrieval_agent = SelfRetrieverAgent(llm)
planner_agent = PlanningAgent(llm)


workflow = StateGraph(State)

# Nodes
workflow.add_node("simplifier_agent", simplifier_agent)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("planner_agent", planner_agent)

# Edges
workflow.set_entry_point("simplifier_agent")
workflow.add_edge("simplifier_agent", "retrieval_agent")
workflow.add_edge("retrieval_agent", "planner_agent")
workflow.add_edge("planner_agent", END)

graph = workflow.compile()
