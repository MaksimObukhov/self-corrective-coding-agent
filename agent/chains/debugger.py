import random
import re
from typing import Final

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from agent.graph.utils.nodes import format_values
from agent.graph.utils.state import State, TestResult

SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
You are an expert debugging agent tasked with analyzing and fixing code that has failed to pass public and private \
tests in a test-driven development environment. Your goal is to identify issues, reflect on the problems, and provide \
debugged code that passes all tests.

## Input Variables:
- <SOLUTION_PLAN>: The original plan for implementing the solution.
- <ORIGINAL_CODE>: The code that failed to pass the tests.
- <FAILED_TESTS>: Detailed information about the failed tests, including test cases and expected outputs.
- <COMPILER_ERROR>: Any compiler or syntax errors encountered (if applicable).

## Your Tasks:
1. Analyze the provided information carefully, focusing on:
   - Given solution plan
   - Compiler errors (if any)
   - Failed test cases
   - Discrepancies between expected and actual outputs
2. Engage in a comprehensive reflection process following these 5 steps:
   <thinking>
   1. Identify the type of error (syntax, logical, runtime, etc.)
   2. Trace the code execution for failed test cases
   3. Compare the implementation against the solution plan
   4. Consider potential edge cases or scenarios not adequately handled
   5. Hypothesize about the root cause of the failures
   </thinking>
3. Develop a debugging strategy based on your reflection.
4. Implement fixes to address the identified issues:
   - Correct any syntax errors
   - Modify logic to handle failed test cases
   - Refactor code if necessary to better align with the problem description and solution plan

## Output Format:
Present your analysis, reflection, debugging strategy and debugged code solution within corresponding tags. \
Write debugged code within <code> tags, do not include ``` blocks. Follow the example format:

## 1. Error Analysis
<analysis>
[Summary of identified errors and their likely causes]
</analysis>

## 2. Reflection on Implementation
<reflection>
[Detailed reflection on how the implementation aligns with the problem description and solution plan]
</reflection>

## 3. Debugging Strategy
<strategy>
[Outline of the approach to fix the identified issues]
</strategy>

## 4. Debugged Code
<code>
[Your debugged code implementation]
</code>
"""

HUMAN_PROMPT_TEMPLATE = """\
1. SOLUTION_PLAN:
<SOLUTION_PLAN>
{SOLUTION_PLAN}
</SOLUTION_PLAN>

2. ORIGINAL_CODE:
<ORIGINAL_CODE>
{ORIGINAL_CODE}
</ORIGINAL_CODE>

3. FAILED_TESTS:
<FAILED_TESTS>
{FAILED_TESTS}
</FAILED_TESTS>

4. COMPILER_ERROR (if any):
<COMPILER_ERROR>
{COMPILER_ERROR}
</COMPILER_ERROR>


Now, analyze the provided <SOLUTION_PLAN>, <ORIGINAL_CODE>, <FAILED_TESTS>, and <COMPILER_ERROR> (if any).
"""


def format_test_result_as_str(test_result: TestResult) -> str:
    formatted_input = format_values(test_result.input)
    formatted_expected = format_values(test_result.expected)
    formatted_actual = format_values(test_result.actual)

    return f"""\
Test Case {test_result.test_index}:
Input:
{formatted_input}
Expected:
{formatted_expected}
Got:
{formatted_actual}
Error: {test_result.error_message}
"""


class DebuggerAgent:
    def __init__(self, llm):
        # self.llm = llm.with_structured_output(...)
        self.llm = llm
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
                HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE),
            ]
        )
        self.runnable = (
                self.prompt
                | self.llm
                | StrOutputParser()
                | (lambda text: re.search(r"<code>(.*?)</code>", text, flags=re.DOTALL).group(1).strip())
        )

    async def __call__(self, state: State) -> dict:
        failed_cases = state.get('test_evaluation_result').debug_info.failed_cases
        if len(failed_cases) > 10:
            failed_cases = random.sample(failed_cases, k=10)
        failed_cases_as_str = '\n\n'.join(format_test_result_as_str(case) for case in failed_cases)

        current_code = state.get('code')

        compiler_error = state.get('test_evaluation_result').debug_info.compile_error_message
        compiler_error = compiler_error if compiler_error is not None else 'No compiler error'

        plans_sorted = state["gen_plans"].plans
        t = state["t_current"]
        solution_plan = f'Algorith name: {plans_sorted[t].algorithm_name}\nPlan: {plans_sorted[t].plan}'
        chain_in = {
            'SOLUTION_PLAN': solution_plan,
            'ORIGINAL_CODE': current_code,
            'FAILED_TESTS': failed_cases_as_str,
            'COMPILER_ERROR': compiler_error,
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.5}).ainvoke(chain_in)
        t_current_update = t+1 if state["k_current"]+1 >= state["k_debug"] else t
        return {
            "code": ai_msg,
            "k_tries": state["k_tries"]+1,
            "k_current": state["k_current"] + 1,
            "t_current": t_current_update,
        }
