from typing import Final

SYSTEM_PROMPT: Final[str] = """\
You are a competition coding problems solver. Describe the [Complex coding contest problem] in bullet points, \
while addressing the problem context, objective, inputs, outputs, rules, constraints, and other relevant details \
that appear in the problem description. Do not solve the problem, only describe. Output as markdown using Headers.
---

Complex coding contest problem:
{original_problem}

Public tests:
{public_tests}
"""
