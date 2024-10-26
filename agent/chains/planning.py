from typing import Final

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

from agent.graph.utils.state import State, PlanningState

SYSTEM_PROMPT_TEMPLATE: Final[str] = """
Given examples of similar problem solutions, generate 3 distinct concrete step-by-step algorithms to solve \
the given problem. Before providing the algorithms, carefully analyze the problem, examples, and test cases to \
develop a comprehensive understanding.

----------------
Important: Think through each algorithm step-by-step before writing it. Clearly break down the logic and approach \
for each distinct solution. The output should be correctly formatted as a XML instance that conforms to the given \
schema below:

<thinking>
  # Analyze the main problem and its requirements
  # Consider the similarities and differences with the example problems
  # Reflect on the test cases and what they reveal about edge cases or specific requirements
  # Brainstorm potential approaches, considering their pros and cons
</thinking>
<algorithm_name> # Name of the algorithm that needs to solve the problem </algorithm_name>
<plan> # Concrete plan to implement the first algorithm as numbered list. </plan>

<thinking>
  # Reflect on the first algorithm and consider how to create a distinct second approach
  # Think about different paradigms or techniques that could be applied
  # Consider trade-offs between time complexity, space complexity, and implementation simplicity
</thinking>
<algorithm_name> # Name of the algorithm that needs to solve the problem </algorithm_name>
<plan> # Concrete plan to implement the second algorithm as numbered list. </plan>

<thinking>
  # Analyze the previous two approaches and brainstorm a third distinct strategy
  # Consider any aspects of the problem not fully addressed by the first two approaches
  # Think about innovative or unconventional methods that could be applied
</thinking>
<algorithm_name> # Name of the algorithm that needs to solve the problem </algorithm_name>
<plan> # Concrete plan to implement the third algorithm as numbered list. </plan>
"""

HUMAN_PROMPT_TEMPLATE = """\
# Examples of similar problem solution:
<examples>
{example_problems}
</examples>

# Problem to be solved: 
<problem>
{simplified_problem}

Test cases:
{public_tests}
</problem>
"""


class PlanningAgent:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(PlanningState)
        # self.llm = llm
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
                HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE),
            ]
        )
        self.runnable = self.prompt | self.llm

    async def __call__(self, state: State) -> dict:
        chain_in = {
            'simplified_problem': state["simplified_problem"],
            'public_tests': state["public_tests"],
            'example_problems': state["example_problems"],
            # 'k': state["k_retrieved"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.3}).ainvoke(chain_in)
        return {"gen_plans": ai_msg}
