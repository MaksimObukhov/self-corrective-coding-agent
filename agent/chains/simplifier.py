from typing import Final
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from agent.graph.utils.state import State, SimplifiedProblemState

SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
You are a world-class competitive programming problems solver. Describe the [Complex coding contest problem] \
in bullet points, while addressing the problem context, objective, inputs, outputs, rules, constraints, and other \
relevant details that appear in the problem description, excluding example explanation. Do not solve the problem, \
only describe. Output as markdown using Headers.
---

Complex coding contest problem:
{original_problem}

Test cases:
{public_tests}
"""

COT = """\
You are a world-class competitive programming problems solver. Describe the [Complex coding contest problem] in \
bullet points, while addressing the problem context, objective, inputs, outputs, rules, constraints, and other \
relevant details that appear in the problem description, excluding example explanation. Do not solve the problem, \
only describe. Output as markdown using Headers.

Before providing the description, carefully analyze the problem and think through its various aspects.

---

Complex coding contest problem:
{original_problem}

Test cases:
{public_tests}

---

Your response should follow this format:

<thinking>
# Analyze the problem statement
- Consider the overall context and theme of the problem
- Identify the key components: inputs, outputs, constraints, and rules
- Reflect on any implicit requirements or challenges
- Think about how the problem relates to common algorithmic paradigms or data structures
- Consider how the test cases inform your understanding of the problem
</thinking>

## Problem Context
<thinking>
- Consider the real-world scenario or abstract setting of the problem
- Think about why this problem might be relevant or interesting
- Reflect on any domain-specific knowledge that might be helpful
</thinking>
- [Bullet points describing the problem context]

## Objective
<thinking>
- Clearly identify the main goal of the problem
- Consider any secondary objectives or implicit goals
- Think about how the objective relates to the problem context
</thinking>
- [Bullet points describing the problem objective]

## Inputs
<thinking>
- Analyze the input format and types
- Consider the range and constraints of each input
- Think about any implicit information provided in the input
</thinking>
- [Bullet points describing the inputs]

## Outputs
<thinking>
- Identify the required output format
- Consider any specific formatting or precision requirements
- Think about how the output relates to the problem objective
</thinking>
- [Bullet points describing the outputs]

## Rules and Constraints
<thinking>
- Identify explicit rules stated in the problem
- Consider implicit rules that might be inferred
- Analyze time and space complexity constraints
- Think about any edge cases or special conditions
</thinking>
- [Bullet points describing rules and constraints]

## Additional Details
<thinking>
- Consider any other relevant information provided in the problem statement
- Think about any clarifications or explanations that might be helpful
- Reflect on how these details might impact the problem-solving approach
</thinking>
- [Bullet points describing any additional relevant details]

Remember, do not solve the problem or explain examples. Focus on providing a clear and comprehensive description \
of the problem itself.
"""


class SimplifierAgent:
    def __init__(self, llm):
        # self.llm = llm.with_structured_output(SimplifiedProblemState)
        self.llm = llm
        self.prompt = PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
        self.runnable = self.prompt | self.llm

    async def __call__(self, state: State) -> dict:
        chain_in = {
            'original_problem': state["original_problem"],
            'public_tests': state["public_tests"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.3}).ainvoke(chain_in)
        return {"simplified_problem": ai_msg.content}
