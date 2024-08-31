from typing import Final
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from agent.graph.utils.state import State, ExampleProblem, ProblemSetState


SYSTEM_PROMPT_TEMPLATE: Final[str] = """\
Given the problem description, provide relevant problems then identify the algorithm behind it and also explain the \
tutorial of the algorithm. Before providing the solutions, think through the problem carefully and consider various approaches.

# Problem description:
{simplified_problem}

# Exemplars:
Recall k={k} relevant and distinct problems (different from problem mentioned above). For each problem:
1. Think about how it relates to the main problem and why it's relevant
2. Consider different approaches to solving the problem
3. Plan out the solution step-by-step
4. Think through the code implementation in {programming_language}
5. Reflect on the underlying algorithm and how to explain it effectively

----------------
Important: Your response must follow the following xml format:
<root>
  <thinking>
    # Analyze the main problem and consider its key characteristics
    # Reflect on similar problems you've encountered and their relevance
    # Consider what makes a problem distinct yet related to the main problem
  </thinking>

  <problem>
    <description> # Describe the problem. </description>
    
    <thinking>
      # Break down the problem into smaller steps
      # Consider different strategies and their pros/cons
      # Think about how this problem relates to the main problem
    </thinking>
    
    <planning> # Planning to solve this problem. </planning>
    
    <thinking>
      # Consider how to implement the solution in {programming_language}
      # Think about potential edge cases or challenges in the implementation
    </thinking>
    
    <code> # Let's think step by step to solve this problem in {programming_language} programming language. </code>
    
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
  
  # if k>1, similarly add more problems here using the same xml tags...
</root>
"""


class SelfRetrieverAgent:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(ProblemSetState)
        self.prompt = PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
        self.runnable = self.prompt | self.llm

    async def __call__(self, state: State) -> dict:
        chain_in = {
            'simplified_problem': state["simplified_problem"],
            'programming_language': state["programming_language"],
            'k': state["k_retrieved"],
        }
        ai_msg = await self.runnable.with_config(configurable={"llm_temperature": 0.5}).ainvoke(chain_in)
        return {"example_problems": ai_msg}

