import asyncio
import sys
import textwrap
from io import StringIO
from typing import List, Dict, Optional
from typing import TypedDict, Literal

from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agent.chains.coding import CodingAgent
from agent.chains.planning import PlanningAgent
from agent.chains.ranking import RankingAgent
from agent.chains.retrieval import SelfRetrieverAgent
from agent.chains.simplifier import SimplifierAgent
from agent.chains.test_gen import TestGenAgent
from agent.graph.utils.state import State, TestCase, TestResult, DebugInfo, TestEvaluationResult
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
ranking_agent = RankingAgent(llm)
test_gen_agent = TestGenAgent(llm)
coding_agent = CodingAgent(llm)


class TestEvaluator:
    def __init__(self, code_string: str):
        self.code_string = textwrap.dedent(code_string.strip())
        self.namespace = {}
        self.compile_error: Optional[str] = None
        self.failed_tests: List[TestResult] = []

    def _prepare_function(self) -> bool:
        """Compile and execute the function code in the namespace."""
        try:
            exec(self.code_string, self.namespace)
            if 'solve' not in self.namespace:
                self.compile_error = "No 'solve' function found in code"
                return False
            return True
        except Exception as e:
            self.compile_error = str(e)
            return False

    def run_test(self, test_case: TestCase, test_index: int) -> TestResult:
        """Run a single test case and return structured result."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Join input lines for the solve function
            input_str = "\n".join(test_case.input)
            result = self.namespace['solve'](input_str)
            output = sys.stdout.getvalue().strip()
            final_output = output or str(result)

            # Join expected output lines for comparison
            expected_output = "\n".join(test_case.output)
            passed = final_output == expected_output

            test_result = TestResult(
                input=input_str,
                expected=expected_output,
                actual=final_output,
                passed=passed,
                test_index=test_index,
                error_message=None
            )

            if not passed:
                self.failed_tests.append(test_result)

            return test_result

        except Exception as e:
            error_msg = str(e)
            test_result = TestResult(
                input="\n".join(test_case.input),
                expected="\n".join(test_case.output),
                actual="",
                passed=False,
                error_message=error_msg,
                test_index=test_index
            )
            self.failed_tests.append(test_result)
            return test_result
        finally:
            sys.stdout = old_stdout

    def evaluate_test_cases(self, test_cases: List[TestCase]) -> TestEvaluationResult:
        """Evaluate test cases and return structured results."""
        compilation_successful = self._prepare_function()

        if not compilation_successful:
            return TestEvaluationResult(
                status=f"Compilation Error: {self.compile_error}",
                all_tests_passed=False,
                compile_error=self.compile_error,
                requires_debugging=True,
                debug_info=DebugInfo(
                    error_type="compilation",
                    error_message=self.compile_error,
                    code=self.code_string
                )
            )

        results = []
        all_passed = True

        for i, test_case in enumerate(test_cases):
            test_result = self.run_test(test_case, i)
            if not test_result.passed:
                all_passed = False
            results.append(f"""
Test Case {i + 1}:
Input:
{test_result.input}
Expected:
{test_result.expected}
Got:
{test_result.actual}
Status: {'PASSED' if test_result.passed else 'FAILED'}
""")

        return TestEvaluationResult(
            status="\n".join(results),
            all_tests_passed=all_passed,
            failed_tests=self.failed_tests,
            requires_debugging=len(self.failed_tests) > 0,
            debug_info=DebugInfo(
                error_type="runtime",
                failed_cases=self.failed_tests,
                code=self.code_string
            ) if self.failed_tests else None
        )


async def test_evaluation_node(state: State) -> Dict:
    """Node function for the langgraph that evaluates the code against test cases."""
    code = state.get("code", "")
    public_tests = state.get("public_tests")
    private_tests = state.get("private_tests")

    # Combine public and private tests if available
    test_cases = [public_tests]
    if private_tests:
        test_cases.append(private_tests)

    evaluator = TestEvaluator(code)
    results = evaluator.evaluate_test_cases(test_cases)

    # Prepare the state update
    state_update = {
        "status": results.status,
        "all_tests_passed": results.all_tests_passed,
    }

    # Add debugging information if there were failures
    if not results.all_tests_passed:
        state_update.update({
            "compile_error": results.compile_error,
            "failed_tests": [test.model_dump() for test in results.failed_tests],
            "requires_debugging": True,
            "debug_info": results.debug_info.model_dump() if results.debug_info else None
        })

    return state_update


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


async def main():
    from dotenv import load_dotenv
    load_dotenv("/Users/maksim/Documents/VSE/Bachelor's thesis/self-corrective-coding-agent/.env")

    default_problem = """\
Problem description. Vipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the \
world. Recently he indulged himself in saving the string population so much that he lost his ability for checking \
brackets (luckily, not permanently ).Being his super-hero friend help him in his time of hardship. Input The first \
line of the input contains an integer T denoting the number of test cases. The description of T test cases follows. \
The first line of each test case contains a single string S denoting the string to be checked. Output For each test \
case, output a single line printing "YES" or "NO" (without " " and in uppercase only) , denoting if the brackets in \
the given string is balanced or not . Constraints 1 ≤ T ≤ 10 1 ≤ length of S ≤ 60 Example Input: 3 ((())) (())() ()(() \
Output: YES YES NO   Explanation Example is self-explanatory.
"""

    default_tests = TestCase(input=["3\n((()))\n(())()\n()(()"], output=["YES\nYES\nNO"])

    input_state = {
        'original_problem': default_problem,
        'public_tests': default_tests,
        'programming_language': 'python',
        'k_retrieved': 3,
        't_debugged': 3,
    }

    graph = workflow.compile()
    final_state = await graph.ainvoke(input_state, debug=True)
    print(final_state)


if __name__ == "__main__":
    asyncio.run(main())
