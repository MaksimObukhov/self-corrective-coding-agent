from typing import Final
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableSerializable

from agent.graph.utils.state import State

SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
You are a world-class competitive programming problems solver. Describe the [Complex coding contest problem] \
in bullet points, while addressing the problem context, objective, inputs, outputs, rules, constraints, and other \
relevant details that appear in the problem description. Do not solve the problem, only describe. \
Output as markdown using Headers.
---

Complex coding contest problem:
{original_problem}

Test cases:
{test_cases}
"""


class Simplifier:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
        self.runnable = self.prompt | self.llm

    async def __call__(self, state: State, ) -> dict:
        chain_in = {
            'original_problem': state["original_problem"],
            'test_cases': state["test_cases"],
        }
        ai_msg = await self.runnable.ainvoke(chain_in)
        return {"simplified_problem": [ai_msg.content]}
