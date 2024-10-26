from typing import Final

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

from agent.graph.utils.state import State, RankingState

# TODO: split into system-human, so that we can cache system
SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
Given the competitive programming problem and the plan to solve the problem in {programming_language} tell whether the \
plan is correct to solve this problem.

----------------
Important: The output should be correctly formatted as a XML instance that conforms to the given schema below:

<plan> {gen_plan} </plan>

<thinking> # Discuss whether the given competitive programming problem is solvable by using the given planning. </thinking>

<confidence> # Confidence score regarding the resolvability of the problem. Must be an integer between 0 and 100. \
</confidence>

"""

HUMAN_PROMPT_TEMPLATE = """\
# Problem: 
{simplified_problem}

# Planning: 
{gen_plan}
"""


class RankingAgent:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(RankingState)
        # self.llm = llm
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
                HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_TEMPLATE),
            ]
        )
        self.runnable = self.prompt | self.llm

    async def __call__(self, state: State) -> dict:
        plans = state["gen_plans"].plans

        chain_in = [
            {
                'simplified_problem': state["simplified_problem"],
                'programming_language': state["programming_language"],
                'gen_plan': plans[i],
            }
            for i in range(len(plans))
        ]
        planning_states_list = await self.runnable.with_config(configurable={"llm_temperature": 0.1}).abatch(chain_in)

        # Step 1: Flatten all plans into a single list
        all_plans = [plan for planning_state in planning_states_list for plan in planning_state.plans]

        # Step 2: Sort the merged plans by confidence in descending order
        sorted_plans = sorted(all_plans, key=lambda x: x.confidence, reverse=True)

        return {"gen_plans": RankingState(plans=sorted_plans), "k_current": 0}
