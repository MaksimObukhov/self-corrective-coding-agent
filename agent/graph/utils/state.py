import operator
from typing import TypedDict, List, Optional, Annotated

from pydantic import BaseModel, Field, conlist


class SimplifiedProblemState(BaseModel):
    """Simplified version of the given complex problem"""

    title: str = Field(..., description="The title of the problem.")
    context: List[str] = Field(..., description="The context or background of the problem.")
    objective: List[str] = Field(...,
                                 description="The main goal or objective of the problem that needs to be achieved.")
    inputs: List[str] = Field(..., description="Details about the input format and what kind of data is expected.")
    outputs: List[str] = Field(...,
                               description="Details about the output format and what kind of results are expected.")
    rules: List[str] = Field(..., description="Specific rules that define how the problem should be solved.")
    constraints: List[str] = Field(..., description="The constraints under which the problem must be solved.")
    relevant_details: List[str] = Field(default=None, description="Other relevant details that appear in the problem.")


class TestCase(BaseModel):
    input: List[str] = Field(..., description="A List of strings representing the input for the test case..")
    output: List[str] = Field(..., description="A List of strings representing the expected output for the test case.")


class Algorithm(BaseModel):
    name: str = Field(..., description="The name of the algorithm used to solve the problem.")
    tutorial: str = Field(..., description="A high-level, generic tutorial about the algorithm.")


class CodeState(BaseModel):
    # todo
    pass


class ExampleProblem(BaseModel):
    description: str = Field(..., description="A detailed description of the problem.")
    code: str = Field(..., description="Step-by-step solution to the problem in the specified programming language.")
    planning: str = Field(..., description="The planning process and approach to solve the problem.")
    algorithm: Algorithm = Field(..., description="The algorithm details used to solve the problem.")


class ProblemSetState(BaseModel):
    problems: conlist(ExampleProblem, max_length=3) = Field(..., description="A list of example problems with \
    a maximum of 3 items.")


class PlanningState(BaseModel):
    plans: List[str] = Field(..., description="The step-by-step plannings process and approach to solve the problem.")
    # prob_score: float | None = Field(None, description="The probability score of the problem.")


default_problem = """\
Problem description. Vipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the \
world. Recently he indulged himself in saving the string population so much that he lost his ability for checking \
brackets (luckily, not permanently ).Being his super-hero friend help him in his time of hardship. Input The first \
line of the input contains an integer T denoting the number of test cases. The description of T test cases follows. \
The first line of each test case contains a single string S denoting the string to be checked. Output For each test \
case, output a single line printing "YES" or "NO" (without " " and in uppercase only) , denoting if the brackets in \
the given string is balanced or not . Constraints 1 ≤ T ≤ 10 1 ≤ length of S ≤ 60 Example Input: 3 ((())) (())() ()(() \
Output: YES YES NO   Explanation Example is self-explanatory.
"""

default_tests = {"input": ["3\n((()))\n(())()\n()(()"], "output": ["YES\nYES\nNO"]}


class State(TypedDict):
    # Input states
    original_problem: str
    public_tests: TestCase
    private_tests: Optional[TestCase]
    programming_language: str
    k_retrieved: int
    t_debugged: int
    runtime_limit: Optional[int]
    status: Optional[str]
    # cf_tags_hidden: List[str]  # If needed, uncomment and add appropriate import for cf_tags_hidden

    # Obtained states during graph workflow
    current_k: Optional[int]
    simplified_problem: Optional[SimplifiedProblemState]
    example_problems: Optional[Annotated[ProblemSetState, operator.add]]
    gen_plans: Optional[List[PlanningState]]
    current_plan: Optional[PlanningState]
    ai_gen_tests: Optional[TestCase]
    code: Optional[CodeState]
