import asyncio
from typing import TypedDict, Literal, Dict

from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agent.chains.coding import CodingAgent
from agent.chains.planning import PlanningAgent
from agent.chains.ranking import RankingAgent
from agent.chains.retrieval import SelfRetrieverAgent
from agent.chains.simplifier import SimplifierAgent
from agent.chains.test_gen import TestGenAgent
from agent.graph.utils.nodes import test_evaluation_node
from agent.graph.utils.state import State, TestCase
from config import CONFIG


# class GraphConfig(TypedDict):
#     model_name: Literal["anthropic", "openai"]


# llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0.3)
model = "gpt-4o-2024-08-06"
# model = "gpt-4o-mini-2024-07-18"
llm = ChatOpenAI(model_name=model, openai_api_key=CONFIG.openai_api_key).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    )
)

# Chains
simplifier_agent = SimplifierAgent(llm)
retrieval_agent = SelfRetrieverAgent(llm)
planner_agent = PlanningAgent(llm)
ranking_agent = RankingAgent(llm)
test_gen_agent = TestGenAgent(llm)
coding_agent = CodingAgent(llm)


workflow = StateGraph(State)

# Nodes
workflow.add_node("simplifier_agent", simplifier_agent)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("planner_agent", planner_agent)
workflow.add_node("ranking_agent", ranking_agent)
workflow.add_node("test_gen_agent", test_gen_agent)
workflow.add_node("coding_agent", coding_agent)
workflow.add_node("test_evaluation", test_evaluation_node)


def next_step(state):
    if state["k_current"] > 3:
        print("Max Iterations done.... Exiting workflow")
        return "max_iterations"
    elif state.get("requires_debugging", False):
        print("Debugging required. Moving to debug_agent")
        return "debug_agent"
    else:
        print("No debugging required. Ending workflow")
        return "ok"


# Edges
workflow.set_entry_point("simplifier_agent")
workflow.add_edge("simplifier_agent", "retrieval_agent")
workflow.add_edge("retrieval_agent", "planner_agent")
workflow.add_edge("planner_agent", "ranking_agent")
workflow.add_edge("ranking_agent", "test_gen_agent")
workflow.add_edge("test_gen_agent", "coding_agent")
workflow.add_edge("coding_agent", "test_evaluation")
workflow.add_edge("test_evaluation", END)

# todo: add check next plan after k>3
# workflow.add_conditional_edges(
#     "test_evaluation", next_step,
#     {"debug_agent": "debug_agent", "ok": END, "max_iterations": END}
# )

graph = workflow.compile()


async def main():
    from dotenv import load_dotenv
    load_dotenv("/Users/maksim/Documents/VSE/Bachelor's thesis/self-corrective-coding-agent/.env")

    def convert_to_test_case(tests_dict: Dict | TestCase) -> TestCase:
        if isinstance(tests_dict, TestCase):
            return tests_dict
        return TestCase(
            input=tests_dict.get("input", []),
            expected_output=tests_dict.get("output", [])
        )

    default_problem = """\
In the year of 30XX participants of some world programming championship live in a single large hotel. The hotel has n floors. Each floor has m sections with a single corridor connecting all of them. The sections are enumerated from 1 to m along the corridor, and all sections with equal numbers on different floors are located exactly one above the other. Thus, the hotel can be represented as a rectangle of height n and width m. We can denote sections with pairs of integers (i, j), where i is the floor, and j is the section number on the floor.\n\nThe guests can walk along the corridor on each floor, use stairs and elevators. Each stairs or elevator occupies all sections (1, x), (2, x), …, (n, x) for some x between 1 and m. All sections not occupied with stairs or elevators contain guest rooms. It takes one time unit to move between neighboring sections on the same floor or to move one floor up or down using stairs. It takes one time unit to move up to v floors in any direction using an elevator. You can assume you don\"t have to wait for an elevator, and the time needed to enter or exit an elevator is negligible.\n\nYou are to process q queries. Each query is a question "what is the minimum time needed to go from a room in section (x_1, y_1) to a room in section (x_2, y_2)?"\n\nInput\n\nThe first line contains five integers n, m, c_l, c_e, v (2 ≤ n, m ≤ 10^8, 0 ≤ c_l, c_e ≤ 10^5, 1 ≤ c_l + c_e ≤ m - 1, 1 ≤ v ≤ n - 1) — the number of floors and section on each floor, the number of stairs, the number of elevators and the maximum speed of an elevator, respectively.\n\nThe second line contains c_l integers l_1, …, l_{c_l} in increasing order (1 ≤ l_i ≤ m), denoting the positions of the stairs. If c_l = 0, the second line is empty.\n\nThe third line contains c_e integers e_1, …, e_{c_e} in increasing order, denoting the elevators positions in the same format. It is guaranteed that all integers l_i and e_i are distinct.\n\nThe fourth line contains a single integer q (1 ≤ q ≤ 10^5) — the number of queries.\n\nThe next q lines describe queries. Each of these lines contains four integers x_1, y_1, x_2, y_2 (1 ≤ x_1, x_2 ≤ n, 1 ≤ y_1, y_2 ≤ m) — the coordinates of starting and finishing sections for the query. It is guaranteed that the starting and finishing sections are distinct. It is also guaranteed that these sections contain guest rooms, i. e. y_1 and y_2 are not among l_i and e_i.\n\nOutput\n\nPrint q integers, one per line — the answers for the queries.\n\nExample\n\nInput\n\n5 6 1 1 3\n2\n5\n3\n1 1 5 6\n1 3 5 4\n3 3 5 3\n\n\nOutput\n\n7\n5\n4\n\nNote\n\nIn the first query the optimal way is to go to the elevator in the 5-th section in four time units, use it to go to the fifth floor in two time units and go to the destination in one more time unit.\n\nIn the second query it is still optimal to use the elevator, but in the third query it is better to use the stairs in the section 2.
"""
    # public_tests = TestCase(input=["3\n((()))\n(())()\n()(()"], output=["YES\nYES\nNO"])
    public_tests = {"input": ["5 6 1 1 3\n2\n5\n3\n1 1 5 6\n1 3 5 4\n3 3 5 3\n"], "output": ["7\n5\n4\n"]}
    private_tests = {"input": ["2 10 1 1 1\n1\n10\n1\n1 5 1 8\n",
                               "4 4 1 0 1\n1\n\n1\n1 2 1 4\n",
                               "2 5 1 0 1\n2\n\n1\n1 4 1 5\n",
                               "10 10 1 8 4\n10\n2 3 4 5 6 7 8 9\n10\n1 1 3 1\n2 1 7 1\n1 1 9 1\n7 1 4 1\n10 1 7 1\n2 1 7 1\n3 1 2 1\n5 1 2 1\n10 1 5 1\n6 1 9 1\n",
                               "2 4 1 1 1\n1\n4\n1\n1 2 1 3\n",
                               "2 2 0 1 1\n\n1\n1\n1 2 2 2\n",
                               "1000 1000 1 1 10\n1\n2\n1\n1 900 1 1000\n",
                               "5 5 1 1 1\n3\n2\n1\n1 5 1 1\n",
                               "2 4 1 1 1\n1\n2\n1\n2 3 2 4\n",
                               "4 4 1 0 1\n4\n\n5\n1 1 2 2\n1 3 2 2\n3 3 4 3\n3 2 2 2\n1 2 2 3\n"],
                     "output": ["3\n",
                                "2\n",
                                "1\n",
                                "3\n4\n4\n3\n3\n4\n3\n3\n4\n3\n",
                                "1\n",
                                "3\n",
                                "100\n",
                                "4\n",
                                "1\n",
                                "6\n4\n3\n5\n4\n"]}

    input_state = {
        "original_problem": default_problem,
        "public_tests": convert_to_test_case(public_tests),
        "private_tests": convert_to_test_case(private_tests),
        "programming_language": "python",
        "k_retrieved": 3,
        "t_debugged": 3,
    }

    final_state = await graph.ainvoke(input_state, debug=True)
    print(final_state.get("debug_info"))


if __name__ == "__main__":
    asyncio.run(main())
