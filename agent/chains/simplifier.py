from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

from agent.graph.utils.state import State, SimplifiedProblemState
from agent.graph.utils.tools import yaml_parser

# TODO:
#  1. add problem_title as variable
#  2. split into system-human, so that we can cache system
SYSTEM_PROMPT_TEMPLATE = """\
You are a world-class competitive programming problems solver. Describe the given [Complex coding contest problem] in \
bullet points, while addressing the problem context, objective, inputs, outputs, rules, constraints, and other \
relevant details that appear in the problem, excluding example explanation.

----------------
Important: Do not solve the problem or explain examples. Focus on providing a clear and comprehensive description \
of the problem itself. Before providing the description, carefully analyze the problem and think through its various \
aspects. Think through each field step-by-step before writing it. The output should be correctly formatted as a XML \
instance that conforms to the given schema below:

  
<title> problem_title from the dataset </title>

<thinking>
# Consider the real-world scenario or abstract setting of the problem
# Think about why this problem might be relevant or interesting
# Reflect on any domain-specific knowledge that might be helpful
</thinking>

<context> # Bullet points describing the problem context </context>

<thinking>
# Clearly identify the main goal of the problem
# Consider any secondary objectives or implicit goals
# Think about how the objective relates to the problem context
</thinking>

<objective> # Bullet points describing the problem objective </objective>

<thinking>
# Analyze the input format and types
# Consider the range and constraints of each input
# Think about any implicit information provided in the input
</thinking>

<inputs> # Bullet points describing the inputs </inputs>

<thinking>
# Identify the required output format
# Consider any specific formatting or precision requirements
# Think about how the output relates to the problem objective
</thinking>

<outputs> # Bullet points describing the outputs </outputs>

<thinking>
# Identify explicit rules stated in the problem
# Consider implicit rules that might be inferred
# Think about any edge cases or special conditions
# Avoid using unicode simbols, use words instead
</thinking>

<rules> # Bullet points describing rules </rules>

<thinking>
# Identify explicit rules stated in the problem
# Consider implicit rules that might be inferred
# Avoid using unicode simbols, use words instead
</thinking>

<constraints> # Bullet points describing constraints </constraints>

<thinking>
# Consider any other relevant information provided in the problem statement
# Think about any clarifications or explanations that might be helpful
# Reflect on how these details might impact the problem-solving approach
</thinking>

<relevant_details> # Bullet points describing any additional relevant details </relevant_details>
"""


HUMAN_PROMPT_TEMPLATE = """\
Complex coding contest problem:
{original_problem}

Test cases:
{public_tests}
"""


class SimplifierAgent:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(SimplifiedProblemState)
        # self.llm = llm
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
                HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE),
            ]
        )
        self.runnable = self.prompt | self.llm | yaml_parser

    async def __call__(self, state: State) -> dict:
        chain_in = {
            'original_problem': state["original_problem"],
            'public_tests': state["public_tests"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.1}).ainvoke(chain_in)
        return {"simplified_problem": ai_msg}
