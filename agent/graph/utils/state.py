import operator
from typing import TypedDict, List, Optional, Annotated, Union

from pydantic import BaseModel, Field, conlist


class SimplifiedProblemState(BaseModel):
    """Simplified version of the given complex problem"""

    title: str = Field(..., description="The title of the problem.")
    context: List[str] = Field(..., description="The context or background of the problem.")
    objective: List[str] = Field(..., description="The main goal or objective of the problem that needs to be "
                                                  "achieved.")
    inputs: List[str] = Field(...,
                              description="Details about the input format and what kind of data is expected.")
    outputs: List[str] = Field(...,
                               description="Details about the output format and what kind of results are expected.")
    rules: List[str] = Field(...,
                             description="Specific rules that define how the problem should be solved.")
    constraints: List[str] = Field(...,
                                   description="The constraints under which the problem must be solved.")
    relevant_details: List[str] = Field(default=None,
                                        description="Other relevant details that appear in the problem.")


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
    plan: str = Field(..., description="The step-by-step plan to solve the problem.")
    code: str = Field(...,
                      description="Step-by-step solution to the problem in the specified programming language.")
    algorithm: Algorithm = Field(...,
                                 description="The algorithm details used to solve the problem.")


class ProblemSetState(BaseModel):
    problems: List[ExampleProblem] = Field(..., description="A list of ExampleProblem.")


class Plan(BaseModel):
    plan: str = Field(..., description="The step-by-step plan to solve the problem.")
    algorithm_name: str = Field(..., description="The name of the algorithm used to solve the problem.")


class PlanRanked(BaseModel):
    plan: str = Field(..., description="The step-by-step plan to solve the problem.")
    algorithm_name: str = Field(..., description="The name of the algorithm used to solve the problem.")
    confidence: int = Field(..., description="The confidence score regarding the resolvability of the problem.")


class PlanningState(BaseModel):
    plans: List[Plan] = Field(..., description="List of Plans.")


class RankingState(BaseModel):
    plans: List[PlanRanked] = Field(..., description="List of ranked plans.")





class TestResult(BaseModel):
    """Model for individual test case results"""
    input: str = Field(..., description="Input that was tested")
    expected: str = Field(..., description="Expected output")
    actual: str = Field(..., description="Actual output received")
    passed: bool = Field(..., description="Whether the test passed")
    error_message: Optional[str] = Field(None, description="Error message if test failed")
    test_index: int = Field(..., description="Index of the test case")


class DebugInfo(BaseModel):
    """Model for debug information"""
    error_type: str = Field(..., description="Type of error (compilation/runtime)")
    error_message: Optional[str] = Field(None, description="Error message if compilation failed")
    failed_cases: Optional[List[TestResult]] = Field(None, description="Details of failed test cases")
    code: str = Field(..., description="Code that was tested")


class TestEvaluationResult(BaseModel):
    """Model for overall test evaluation results"""
    status: str = Field(..., description="Overall test execution status")
    all_tests_passed: bool = Field(..., description="Whether all tests passed")
    compile_error: Optional[str] = Field(None, description="Compilation error if any")
    failed_tests: List[TestResult] = Field(default_factory=list, description="List of failed tests")
    requires_debugging: bool = Field(..., description="Whether debugging is needed")
    debug_info: Optional[DebugInfo] = Field(None, description="Debug information if needed")


class State(TypedDict):
    """
    Represents the complete state of the problem-solving workflow.

    Input States:
    - Initialized at the start of the workflow

    Workflow States:
    - Updated during the execution of the workflow
    """

    # Input states - Required at initialization
    original_problem: str  # Original problem text/description
    public_tests: TestCase  # Public test cases for validation
    programming_language: str  # Target programming language
    k_retrieved: int  # Maximum number of examples to retrieve
    t_debugged: int  # Maximum number of debug attempts

    # Input states - Optional at initialization
    private_tests: Optional[TestCase]  # Additional private test cases
    runtime_limit: Optional[int]  # Maximum runtime allowed
    status: Optional[str]  # Current status of the workflow

    # Workflow states - Updated during execution
    k_current: Optional[int]  # Current number of examples retrieved
    t_current: Optional[int]  # Current number of debug attempts
    simplified_problem: Optional[SimplifiedProblemState]  # Processed problem definition
    example_problems: Optional[Annotated[ProblemSetState, operator.add]]  # Retrieved similar examples
    gen_plans: Optional[Union[PlanningState, RankingState]]  # Generated solution plans
    current_plan: Optional[Union[PlanningState, RankingState]]  # Currently active plan
    ai_gen_tests: Optional[TestCase]  # AI-generated test cases
    code: Optional[str]  # Generated solution code
