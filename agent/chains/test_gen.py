import re
from typing import Final

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

from agent.graph.utils.state import State

SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
You are an expert software engineer tasked with creating comprehensive unit tests for a given problem before \
implementing the solution. Your goal is to generate a set of unit tests that will guide the development process and \
ensure the final implementation meets all requirements.

## Input Variables:
- <PROBLEM_DESCRIPTION>: A detailed description of the problem to be solved.
- <SOLUTION_PLAN>: An outline of the proposed solution approach.
- <PUBLIC_TESTS>: Any pre-existing public tests that must be passed.

## Your Tasks:
1. Analyze the <PROBLEM_DESCRIPTION> and <SOLUTION_PLAN> carefully.
2. Identify key functionalities and edge cases that need to be tested.
3. Create a comprehensive set of unit tests that cover:
  - Basic functionality
  - Edge cases
  - Input validation
  - Expected outputs
  - Error handling
4. Ensure your tests cover all aspects mentioned in the <SOLUTION_PLAN>.
5. Incorporate any <PUBLIC_TESTS> provided into your test suite.
6. For each test, provide:
  - A detailed thinking process explaining how you arrived at the test case within <thinking> tags
  - A descriptive name
  - Input values
  - Expected output or behavior
  - A brief explanation of what the test is checking
7. Use a consistent format for presenting the tests, preferably in a way that can be easily translated into actual unit test code.

Remember, these tests will be used to drive the development of the actual solution, so they should be thorough and \
cover all possible scenarios outlined in the <PROBLEM_DESCRIPTION> and <SOLUTION_PLAN>.

## Output Format:
Present your tests in a clear, organized manner. Use markdown formatting for readability. For example:

```markdown
<thinking>
# Before writing actual test cases, thoroughly consider each potential test case type, including edge cases and its \
necessary variations.
</thinking>

### Test 1: [Test Name]
<thinking>
# Consider the problem requirements
# Analyze the solution plan
# Formulate a clear description of the test's purpose
# Determine critical functionality to test
# Design input to test this functionality
# Identify potential edge cases
# Predict expected output based on correct implementation
</thinking>

- Input: [Specify input]
- Expected output: [Specify expected output]
- Explanation: [Brief explanation of what the test is checking]

### Test 2: [Test Name]
...
```
"""

HUMAN_PROMPT_TEMPLATE: Final[str] = """\
1. PROBLEM_DESCRIPTION:
<PROBLEM_DESCRIPTION>
{PROBLEM_DESCRIPTION}
</PROBLEM_DESCRIPTION>

2. SOLUTION_PLAN:
<SOLUTION_PLAN>
{SOLUTION_PLAN}
</SOLUTION_PLAN>

3. PUBLIC_TESTS:
<PUBLIC_TESTS>
{PUBLIC_TESTS}
</PUBLIC_TESTS>

Now, generate a comprehensive set of unit tests based on the provided <PROBLEM_DESCRIPTION>, <SOLUTION_PLAN>, and \
<PUBLIC_TESTS>. Use the detailed thinking process to explain your reasoning for each test. Also include the given <PUBLIC_TESTS>.
"""


# we want to pass only 1 k-th plan, simplififed problem, public tests
class TestGenAgent:
    def __init__(self, llm):
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
                | (lambda text: re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL))
        )

    async def __call__(self, state: State) -> dict:
        plans_sorted = state["gen_plans"].plans
        i = state["k_current"]
        chain_in = {
            'PROBLEM_DESCRIPTION': state["simplified_problem"],
            'SOLUTION_PLAN': plans_sorted[i],
            'PUBLIC_TESTS': state["public_tests"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.5}).ainvoke(chain_in)
        return {"ai_gen_tests": ai_msg}
