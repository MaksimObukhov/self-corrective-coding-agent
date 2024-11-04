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
You are an experienced programming contestant tasked with implementing a single function that solves the given problem \
based on the unit tests and the solution plan.

## Input Components:
- <PROBLEM>: Simplified problem description
- <SOLUTION_PLAN>: Pseudocode describing the algorithmic approach
- <UNIT_TESTS>: Test cases with input parameters and expected outputs

## Implementation Requirements:
Create exactly ONE function 'solve' that:
- Takes input parameters as manual input
- Implements the algorithm described in the solution plan
- Prints the result (print or use I/O)

## Output Format:
Present your solution in a clear, organized manner. Provide explanations before code sections within <thinking> tags,
write code within <code> tags then. Follow the example format:
<thinking>
# Analyze relevant tests
# Consider edge cases from tests
# Describe the main purpose of this component
# Decide on necessary data structures
# Think about efficiency and potential optimizations
</thinking>

<code>
# Example code
def solve(input_str: str) -> str:
    data = input_str.splitlines()
    T = int(data[0])
    results = []

    index = 1
    for _ in range(T):
        N, K = map(int, data[index].split())
        if N == 0:
            results.append(0)
            index += 1
            continue
        arr = list(map(int, data[index + 1].split()))

        dp = [[False] * (K + 1) for _ in range(N + 1)]
        dp[0][0] = True  # 0 sum is always possible with 0 elements

        for i in range(1, N + 1):
            dp[i][0] = True  # 0 sum is possible with any number of elements
            for j in range(1, K + 1):
                if arr[i - 1] <= j:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
                else:
                    dp[i][j] = dp[i - 1][j]

        results.append(1 if dp[N][K] else 0)
        index += 2

    print('\n'.join(map(str, results)))
</code>

## Important Notes:
- Generate one function 'solve' that takes 'input_str' as input within <code> tags, without any comments or test cases
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
{AI_GEN_TESTS}
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
        t = state["t_current"]
        solution_plan = f'Algorith name: {plans_sorted[t].algorithm_name}\nPlan: {plans_sorted[t].plan}'
        chain_in = {
            'PROBLEM': state["simplified_problem"],
            'PROGRAMMING_LANGUAGE': state["programming_language"],
            'SOLUTION_PLAN': solution_plan,
            'AI_GEN_TESTS': state["ai_gen_tests"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.1}).ainvoke(chain_in)
        return {"code": ai_msg}
