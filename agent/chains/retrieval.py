from typing import Final

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate

from agent.graph.utils.state import State

# TODO: split into system-human, so that we can cache system
SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
Given the problem description, provide relevant problems then identify the algorithm behind it and also explain the \
tutorial of the algorithm.

# Exemplars:
Recall {k} relevant and distinct problems (different from problem mentioned above). For each problem:
1. Think about how it relates to the main problem and why it's relevant
2. Consider different approaches to solving the problem
3. Plan out the solution step-by-step as numbered list
4. Think through the code implementation in {programming_language}
5. Reflect on the underlying algorithm and how to explain it effectively

----------------
Important: Before providing the solutions, think through the problem carefully and consider various approaches. \
The output should be correctly formatted as a XML instance that conforms to the given schema below:

<thinking>
# Analyze the main problem and consider its key characteristics
# Reflect on similar problems you've encountered and their relevance
# Consider what makes a problem distinct yet related to the main problem
</thinking>

<problem>
  <description> # Describe the problem. </description>
  
  <thinking>
    # Break down the problem into smaller steps as numbered list
    # Consider different strategies and their pros/cons
    # Think about how this problem relates to the main problem
  </thinking>
  
  <plan> # The step-by-step plan to solve the problem as numbered list. </plan>
  
  <thinking>
    # Consider how to implement the solution in {programming_language}
    # Think about potential edge cases or challenges in the implementation
  </thinking>
  
  <code> # Code to solve this problem in {programming_language} programming language. </code>
  
  <thinking>
    # Reflect on the underlying algorithm used in the solution
    # Consider how to explain this algorithm in a clear and concise manner
    # Think about common applications and variations of this algorithm
  </thinking>
  
  <algorithm>
    <name> # Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, 
    Recursive, Binary search, and so on) that needs to be used to solve the original problem. </name>
    <tutorial> # Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial 
    for solving this types of problem. Do not generate code. </tutorial>
  </algorithm>
</problem>

# similarly add 2 more problems here
<problem></problem>
<problem></problem>
"""

HUMAN_PROMPT_TEMPLATE = """\
# Problem description:
{simplified_problem}
"""


# TODO: write simple regex extracting problems and deleting thinkings
class SelfRetrieverAgent:
    def __init__(self, llm):
        # self.llm = llm.with_structured_output(ProblemSetState)
        self.llm = llm
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
            'programming_language': state["programming_language"],
            'k': state["k_retrieved"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.3}).ainvoke(chain_in)
        return {"example_problems": ai_msg}

