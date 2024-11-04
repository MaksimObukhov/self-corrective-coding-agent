import sys
import textwrap
import traceback
from dataclasses import asdict
from io import StringIO
from typing import List, Dict, Optional

from agent.graph.utils.state import State, TestCase, TestResult, DebugInfo, TestEvaluationResult


def format_values(value: str) -> str:
    split_values = value.split()
    if len(split_values) > 30:
        return " ".join(split_values[:15]) + " ... " + " ".join(split_values[-15:])
    return " ".join(split_values)


class TestEvaluator:
    def __init__(self, solve_function_str: str):
        """
        Initialize the evaluator with the solve function as a string.
        This allows us to catch compilation errors.
        """
        self.solve_function_str = solve_function_str
        self.namespace = {}
        self.failed_tests: List[TestResult] = []
        self.compile_error: Optional[str] = None

    def _prepare_function(self) -> bool:
        """
        Compile and execute the function code in the namespace.
        Returns True if successful, False if there's a compilation error.
        """
        try:
            # Try to compile and execute the function code
            exec(self.solve_function_str, self.namespace)
            if 'solve' not in self.namespace:
                self.compile_error = "No 'solve' function found in code"
                return False
            return True
        except Exception as e:
            self.compile_error = str(e)
            return False

    def run_test(self, input_str: str, expected_output: str, test_index: int) -> TestResult:
        """Run a single test case and return structured result."""
        original_stdin = sys.stdin
        original_stdout = sys.stdout

        sys.stdin = StringIO(input_str)
        test_output = StringIO()
        sys.stdout = test_output

        try:
            # Run the solve function
            self.namespace['solve'](input_str)
            actual_output = test_output.getvalue().strip()
            expected_output = expected_output.strip()

            # Create test result
            test_result = TestResult(
                input=input_str,
                expected=expected_output,
                actual=actual_output,
                passed=" ".join(expected_output.strip().split()) == " ".join(actual_output.strip().split()),
                test_index=test_index
            )

            if not test_result.passed:
                self.failed_tests.append(test_result)

            return test_result

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            test_result = TestResult(
                input=input_str,
                expected=expected_output,
                actual="",
                passed=False,
                test_index=test_index,
                error_message=error_msg
            )
            self.failed_tests.append(test_result)
            return test_result

        finally:
            sys.stdin = original_stdin
            sys.stdout = original_stdout

    def evaluate_test_cases(self, test_cases: List[TestCase]) -> TestEvaluationResult:
        """
        Evaluate test cases and return structured results with debug information.
        """
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

        for i, test_case in enumerate(test_cases):
            test_result = self.run_test(
                test_case.input,
                test_case.expected_output,
                i
            )

            if not test_result.passed:
                all_passed = False

            # Format Input, Expected, and Got fields
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
