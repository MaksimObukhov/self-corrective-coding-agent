from typing import Final

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from agent.graph.utils.state import State

SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
You are an expert debugging agent tasked with analyzing and fixing code that has failed to pass public and private \
tests in a test-driven development environment. Your goal is to identify issues, reflect on the problems, and provide \
debugged code that passes all tests.

## Input Variables:
- <ORIGINAL_CODE>: The code that failed to pass the tests.
- <FAILED_TESTS>: Detailed information about the failed tests, including test cases and expected outputs.
- <COMPILER_ERROR>: Any compiler or syntax errors encountered (if applicable).
- <SOLUTION_PLAN>: The original plan for implementing the solution.

## Your Tasks:
1. Analyze the provided information carefully, focusing on:
   - Compiler errors (if any)
   - Failed test cases
   - Discrepancies between expected and actual outputs
2. Engage in a comprehensive reflection process:
   <thinking>
   1. Identify the type of error (syntax, logical, runtime, etc.)
   2. Trace the code execution for failed test cases
   3. Compare the implementation against the problem description and solution plan
   4. Consider potential edge cases or scenarios not adequately handled
   5. Evaluate if the original solution approach is fundamentally sound or needs revision
   6. Hypothesize about the root cause of the failures
   </thinking>
3. Develop a debugging strategy based on your reflection.
4. Implement fixes to address the identified issues:
   - Correct any syntax errors
   - Modify logic to handle failed test cases
   - Refactor code if necessary to better align with the problem description and solution plan

## Output Format:
Present your analysis, reflection, and debugged solution in a clear, organized manner. Use markdown formatting for \
readability. For example:

# Debugging Analysis and Solution

## 1. Error Analysis
<thinking>
[Detailed thought process about the errors encountered]
</thinking>

[Summary of identified errors and their likely causes]

## 2. Failed Tests Analysis
<thinking>
[Detailed thought process about why specific tests failed]
</thinking>

[Summary of insights from analyzing failed tests]

## 3. Reflection on Implementation
<thinking>
[Detailed reflection on how the implementation aligns with the problem description and solution plan]
</thinking>

[Summary of key insights from the reflection]

## 4. Debugging Strategy
[Outline of the approach to fix the identified issues]

## 5. Debugged Code

```{PROGRAMMING_LANGUAGE}
[Your debugged code implementation]
```
"""

HUMAN_PROMPT_TEMPLATE = """\
1. ORIGINAL_CODE:
<ORIGINAL_CODE>
{ORIGINAL_CODE}
</ORIGINAL_CODE>

2. FAILED_TESTS:
<FAILED_TESTS>
{FAILED_TESTS}
</FAILED_TESTS>

3. COMPILER_ERROR (if any):
<COMPILER_ERROR>
{COMPILER_ERROR}
</COMPILER_ERROR>

4. SOLUTION_PLAN:
<SOLUTION_PLAN>
{SOLUTION_PLAN}
</SOLUTION_PLAN>

Now, analyze the provided <ORIGINAL_CODE>, <FAILED_TESTS>, <COMPILER_ERROR> (if any), and <SOLUTION_PLAN>. Provide a \
comprehensive debugging analysis and a corrected implementation in {PROGRAMMING_LANGUAGE} that should pass all tests.
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
        self.runnable = self.prompt | self.llm

    async def __call__(self, state: State) -> dict:
        plans_sorted = state["gen_plans"].plans
        i = state["k_current"]
        chain_in = {
            'simplified_problem': state["simplified_problem"],
            'public_tests': state["public_tests"],
            'programming_language': state["programming_language"],
            'gen_plan': plans_sorted[i],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.3}).ainvoke(chain_in)
        return {"ai_gen_tests": ai_msg}
