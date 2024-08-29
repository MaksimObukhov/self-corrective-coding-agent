from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages


class TestCase(TypedDict):
    inputs: str
    outputs: str


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    original_problem: str
    test_cases: list[TestCase]
    simplified_problem: str
    runtime_limit: int
    status: str
