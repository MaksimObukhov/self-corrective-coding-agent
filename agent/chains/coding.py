import re
from typing import Final

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from agent.graph.utils.state import State

"""\
You are an expert software engineer tasked with implementing a code solution based on a set of unit tests and a \
solution plan. Your goal is to write a function in {PROGRAMMING_LANGUAGE} based in the outlined solution approach and \
the provided tests cases.

## Input Variables:
- <SOLUTION_PLAN>: An outline of the proposed solution approach.
- <UNIT_TESTS>: A comprehensive set of unit tests generated for this problem.

## Your Tasks:
1. Carefully analyze the <SOLUTION_PLAN>, and <UNIT_TESTS>.
2. Implement the solution in {PROGRAMMING_LANGUAGE}, following these steps:
  a. Engage in a detailed thinking process to plan your implementation.
  b. Review the relevant unit tests.
  c. Code a single function that solves the problem.
3. Ensure your implementation:
  - Follows the approach outlined in the <SOLUTION_PLAN>
  - Passes all provided <UNIT_TESTS>
  - Handles edge cases and error conditions as specified in the tests
4. Use clear variable names, add comments where necessary, and follow best practices for {PROGRAMMING_LANGUAGE}.

## Output Format:
Present your solution in a clear, organized manner. Use markdown code blocks for the implementation and provide \
explanations before code sections. For example:

# Solution Implementation
<thinking>
1. Analyze relevant tests
2. Consider edge cases from tests
3. Describe the main purpose of this component
4. Decide on necessary data structures
5. Think about efficiency and potential optimizations
</thinking>

<code>
```{PROGRAMMING_LANGUAGE}
[Your code implementation]
```
</code>
"""

SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
You are an expert software engineer tasked with implementing a single function that solves a programming problem based \
on unit tests and a solution plan. Your generated function will be tested using assert statements against test cases.

## Input Components:
- <PROBLEM>: Simplified problem description
- <SOLUTION_PLAN>: Pseudocode describing the algorithmic approach
- <UNIT_TESTS>: Test cases with input parameters and expected outputs

## Implementation Requirements:
1. Create exactly ONE function that:
   - Takes input parameters as specified in the unit tests
   - Implements the algorithm described in the solution plan
   - Returns the result (do not print or use I/O)
   - Named as 'solve'
2. Function must be pure and self-contained:
   - No global variables or side effects
   - No input/output operations
   - No reading from stdin or printing to stdout

## Output Format:
Present your solution in a clear, organized manner. Use markdown code blocks for the implementation and provide \
explanations before code sections. For example:

```markdown
# Solution Implementation
<thinking>
# Analyze relevant tests
# Consider edge cases from tests
# Describe the main purpose of this component
# Decide on necessary data structures
# Think about efficiency and potential optimizations
</thinking>

<code>
def solve(n: int, k: int, arr: List[int]) -> int:
    dp = [0] * (k + 1)
    dp[0] = 1
    for i in arr:
        for j in range(k - i, -1, -1):
            if dp[k]:
                break
            if dp[j]:
                dp[j + i] = 1
    return dp[k]
</code>
```

## Important Notes:
- Generate ONLY the function, without any imports, test cases, or main blocks
- Function must be directly testable using assert statements
- No print statements or I/O operations allowed
- All necessary processing must happen within the function
"""

HUMAN_PROMPT_TEMPLATE: Final[str] = """\
1. PROBLEM:
<PROBLEM>
{PROBLEM}
</PROBLEM>

2. SOLUTION_PLAN:
<SOLUTION_PLAN>
{SOLUTION_PLAN}
</SOLUTION_PLAN>

3. UNIT_TESTS:
<UNIT_TESTS>
{UNIT_TESTS}
</UNIT_TESTS>

Now, implement the solution in {PROGRAMMING_LANGUAGE} based on the provided <PROBLEM>, <SOLUTION_PLAN> and <UNIT_TESTS>.
"""


class CodingAgent:
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
        plans_sorted = state["gen_plans"].plans
        i = state["k_current"]
        solution_plan = f'Algorith name: {plans_sorted[i].algorithm_name}\nPlan: {plans_sorted[i].plan}'
        chain_in = {
            'PROBLEM': state["simplified_problem"],
            'PROGRAMMING_LANGUAGE': state["programming_language"],
            'SOLUTION_PLAN': solution_plan,
            'UNIT_TESTS': state["ai_gen_tests"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.1}).ainvoke(chain_in)
        return {"code": ai_msg}
