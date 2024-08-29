from typing import Final
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT: Final[str] = """\
Given the problem, provide relevant problems then identify the algorithm behind it and \
also explain the tutorial of the algorithm.

# Problem:
{original_problem}

# Exemplars:
Recall k relevant and distinct problems (different from problem mentioned above). For each problem:
1. describe it
2. generate {language} code step by step to solve that problem
3. finally generate a planning to solve that problem

# Algorithm:
----------------
Important:
Your response must follow the following xml format:
<root>
  <problem>
  # Recall k relevant and distinct problems (different from problem mentioned above). \
  Write each problem in the following format.
    <description> # Describe the problem. </description>
    <code> # Let's think step by step to solve this problem in {language} programming language. </code>
    <planning> # Planning to solve this problem. </planning>
  </problem>
  # similarly add more problems here...
  <algorithm>
  # Identify the algorithm (Brute-force, Dynamic Programming, Divide-and-conquer, Greedy, Backtracking, \
  Recursive, Binary search, and so on) that needs to be used to solve the original problem.
  # Write a useful tutorial about the above mentioned algorithms. Provide a high level generic tutorial \
  for solving this types of problem. Do not generate code.
  </algorithm>
</root>
"""

def create_retrieval_chain():


    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    
    llm = ChatOpenAI(model_name="claude-3-sonnet-20240229", temperature=0)
    
    return prompt | llm | StrOutputParser()

