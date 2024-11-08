import sys
import threading
import traceback
from io import StringIO
from typing import List, Dict, Optional

from agent.graph.utils.state import State, TestCase, TestResult, DebugInfo, TestEvaluationResult


def format_values(value: str) -> str:
    if len(value) > 80:
        return value[:40] + " ... " + value[-40:]
    return value


class TimeoutException(Exception):
    pass


def run_with_timeout(func, args, timeout_seconds=3):
    """Run a function with a timeout using threading."""
    result = {'output': None, 'exception': None}

    def worker():
        try:
            output = func(*args)
            result['output'] = output
        except Exception as e:
            result['exception'] = traceback.format_exc()

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        return {'exception': "Time Limit Exceeded (3 seconds), most likely due to infinite loop."}
    return result


def execute_solution(input_str: str, solve_function_str: str) -> str:
    """Execute the solution in a controlled environment and return the output string."""
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    test_output = StringIO()
    sys.stdout = test_output

    try:
        # Create a new namespace for execution
        namespace = {}
        exec(solve_function_str, namespace)

        if 'solve' not in namespace:
            raise Exception("No 'solve' function found in code")

        # Set up input
        sys.stdin = StringIO(input_str)

        # Run the solution
        namespace['solve'](input_str)

        # Get the output
        return test_output.getvalue()

    finally:
        # Restore stdout
        sys.stdout = old_stdout


class TestEvaluator:
    def __init__(self, solve_function_str: str):
        self.solve_function_str = solve_function_str
        self.namespace = {}
        self.failed_tests: List[TestResult] = []
        self.compile_error: Optional[str] = None

    def _prepare_function(self) -> bool:
        """Compile and execute the function code in the namespace."""
        try:
            exec(self.solve_function_str, self.namespace)
            if 'solve' not in self.namespace:
                self.compile_error = "No 'solve' function found in code"
                return False
            return True
        except Exception as e:
            self.compile_error = str(e)
            return False

    def run_test(self, input_str: str, expected_output: str, test_index: int) -> TestResult:
        """Run a single test case with timeout."""
        result = run_with_timeout(
            execute_solution,
            args=(input_str, self.solve_function_str),
            timeout_seconds=1
        )

        if result['exception'] is not None:
            error_msg = result['exception']
            test_result = TestResult(
                input=input_str,
                expected=expected_output.strip(),
                actual="",
                passed=False,
                test_index=test_index,
                error_message=error_msg
            )
            self.failed_tests.append(test_result)
            return test_result

        actual_output = result['output']
        if actual_output is None:
            error_msg = "No output produced"
            test_result = TestResult(
                input=input_str,
                expected=expected_output.strip(),
                actual="",
                passed=False,
                test_index=test_index,
                error_message=error_msg
            )
            self.failed_tests.append(test_result)
            return test_result

        # Create test result with proper string comparison
        test_result = TestResult(
            input=input_str,
            expected=expected_output.strip(),
            actual=actual_output.strip(),
            passed=" ".join(expected_output.strip().split()) == " ".join(actual_output.strip().split()),
            test_index=test_index
        )

        if not test_result.passed:
            self.failed_tests.append(test_result)
        return test_result

    def evaluate_test_cases(self, test_cases: List[TestCase]) -> TestEvaluationResult:
        """Evaluate all test cases and return structured results."""
        try:
            # First check if the function compiles
            compilation_successful = self._prepare_function()

            if not compilation_successful:
                return TestEvaluationResult(
                    status=f"Compilation Error: {self.compile_error}",
                    all_tests_passed=False,
                    compile_error=self.compile_error,
                    requires_debugging=True,
                    debug_info=DebugInfo(
                        error_type="compilation",
                        compile_error_message=self.compile_error,
                        code=self.solve_function_str,
                    )
                )

            results = []
            all_passed = True
            timeout_occurred = False
            error_message = None

            for i, test_case in enumerate(test_cases):
                try:
                    test_result = self.run_test(
                        test_case.input,
                        test_case.expected_output,
                        i
                    )

                    if test_result.error_message and "Time Limit Exceeded" in test_result.error_message:
                        timeout_occurred = True
                        error_message = f"Test case {i + 1} timed out after 5 seconds"
                        break

                    if not test_result.passed:
                        all_passed = False

                    # Format values for display
                    formatted_input = format_values(test_result.input)
                    formatted_expected = format_values(test_result.expected)
                    formatted_actual = format_values(test_result.actual)

                    results.append(f"""\
Test Case {i + 1}:
Input:
{formatted_input}
Expected:
{formatted_expected}
Got:
{formatted_actual}
Status: {'PASSED' if test_result.passed else 'FAILED'}
""" + (f"Error:\n{test_result.error_message}" if test_result.error_message else ""))

                except Exception as e:
                    error_message = f"Error in test case {i + 1}: {str(e)}\n{traceback.format_exc()}"
                    all_passed = False
                    break

            if timeout_occurred:
                return TestEvaluationResult(
                    status=error_message,
                    all_tests_passed=False,
                    requires_debugging=True,
                    debug_info=DebugInfo(
                        error_type="runtime",
                        compile_error_message="Timeout exception, most likely due to infinity loop.",
                        failed_cases=self.failed_tests,
                        code=self.solve_function_str,
                    )
                )

            return TestEvaluationResult(
                status="\n".join(results),
                all_tests_passed=all_passed,
                requires_debugging=len(self.failed_tests) > 0,
                debug_info=DebugInfo(
                    error_type="runtime",
                    compile_error_message=self.compile_error,
                    failed_cases=self.failed_tests,
                    code=self.solve_function_str,
                ) if self.failed_tests else None
            )

        except Exception as e:
            error_message = f"Unexpected error during test evaluation: {str(e)}\n{traceback.format_exc()}"
            return TestEvaluationResult(
                status=error_message,
                all_tests_passed=False,
                requires_debugging=True,
                debug_info=DebugInfo(
                    error_type="runtime",
                    compile_error_message=error_message,
                    failed_cases=self.failed_tests,
                    code=self.solve_function_str,
                )
            )


async def test_evaluation_node(state: State) -> Dict:
    """Node function for the langgraph that evaluates the code against test cases."""

    code = state.get("code", "")
    public_tests = state.get("public_tests", [])
    private_tests = state.get("private_tests", [])
    all_test_cases = []

    # Process test cases
    if not public_tests:
        raise ValueError("No public test cases found")

    all_test_cases.extend(public_tests)
    if private_tests:
        all_test_cases.extend(private_tests)

    # Create evaluator and run tests
    evaluator = TestEvaluator(code)
    results = evaluator.evaluate_test_cases(all_test_cases)

    # Prepare the state update with additional error handling
    state_update = {
        "status": results.status,
        "all_tests_passed": results.all_tests_passed,
    }

    # Add debugging information if there were failures
    if not results.all_tests_passed:
        debug_info = results.debug_info.model_dump() if results.debug_info else None
        state_update.update({
            "requires_debugging": True,
            "debug_info": debug_info
        })

    return {"test_evaluation_result": TestEvaluationResult(**state_update)}


def main():
    def convert_to_test_case(tests_dict: Dict, is_public: bool = False) -> List[TestCase]:
        return [
            TestCase(input=inp, expected_output=out, is_public=is_public)
            for inp, out in zip(tests_dict.get('input', []), tests_dict.get('output', []))
        ]

    code = """\
def solve(input_str: str):
    import sys
    from collections import deque, defaultdict

    input_data = input_str.strip().split('\\n')
    n, m = map(int, input_data[0].split())

    # Adjacency list for roads
    roads = defaultdict(lambda: [[], []])  # roads[town] = [pedestrian_roads, bike_roads]

    for i in range(1, m + 1):
        vi, ui, ti = map(int, input_data[i].split())
        roads[vi][ti].append(ui)

    # BFS setup
    queue = deque([(1, 0)])  # (current_town, current_road_type)
    max_length = defaultdict(lambda: [0, 0])  # max_length[town] = [max_length_pedestrian, max_length_bike]
    max_length[1][0] = 1  # Start from town 1 with a pedestrian road

    # BFS loop
    while queue:
        current_town, current_road_type = queue.popleft()
        next_road_type = 1 - current_road_type  # Flip road type

        for next_town in roads[current_town][current_road_type]:
            if max_length[next_town][next_road_type] < max_length[current_town][current_road_type] + 1:
                max_length[next_town][next_road_type] = max_length[current_town][current_road_type] + 1
                if max_length[next_town][next_road_type] > 10**18:
                    print(-1)
                    return
                queue.append((next_town, next_road_type))

    # Find the maximum length
    result = max(max_length[town][road_type] for town in range(1, n + 1) for road_type in [0, 1])
    print(result - 1)  # Subtract 1 because we started counting from 1
"""
    p_test = convert_to_test_case({'input': ['2 2\n1 2 0\n2 2 1\n', '2 3\n1 2 0\n2 2 1\n2 2 0\n'],
                                   'output': ['3', '-1\n']})
    priv_test = convert_to_test_case({'input': ['1 2\n1 1 0\n1 1 1\n',
                                                '3 13\n1 3 1\n1 1 0\n1 1 1\n2 3 1\n2 2 0\n3 2 1\n3 1 0\n1 2 1\n2 1 0\n1 3 0\n2 3 0\n2 2 1\n3 3 0\n',
                                                '5 0\n',
                                                '1 1\n1 1 0\n',
                                                '3 6\n3 1 1\n1 2 1\n2 1 1\n1 3 0\n3 2 1\n2 2 0\n',
                                                '2 2\n2 1 1\n1 2 0\n'],
                                      'output': ['-1\n', '-1\n', '0', '1', '30', '2']})

    all_tests = p_test + priv_test

    evaluator = TestEvaluator(code)
    results = evaluator.evaluate_test_cases(all_tests)
    print(results)


if __name__ == "__main__":
    # Ensure the main entry point is guarded
    main()
