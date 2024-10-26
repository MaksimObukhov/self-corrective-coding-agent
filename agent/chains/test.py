import json
import yaml

# The JSON string
json_data = '''
{
  "output": {
    "title": "Bracket Balance Checker",
    "context": [
      "Vipul is a superhero who ensures the bracket ratio of strings is maintained.",
      "He has temporarily lost the ability to check brackets.",
      "The user is tasked with helping him by checking whether the brackets in given strings are balanced."
    ],
    "objective": [
      "Determine if the brackets in each provided string are balanced.",
      "Output 'YES' if balanced, 'NO' otherwise."
    ],
    "inputs": [
      "An integer T denoting the number of test cases.",
      "For each test case, a string S that contains only brackets."
    ],
    "outputs": [
      "For each test case, output 'YES' if the string has balanced brackets, otherwise output 'NO'."
    ],
    "rules": [
      "A string is considered balanced if every opening bracket '(' has a corresponding closing bracket ')'.",
      "The order of brackets must also be valid; i.e., no closing bracket should appear before its matching opening bracket."
    ],
    "constraints": [
      "1 ≤ T ≤ 10",
      "1 ≤ length of S ≤ 60"
    ],
    "relevant_details": [
      "The problem is focused on checking parentheses, which is a common problem in programming and algorithms."
    ]
  }
}
'''

# Parse the JSON string into a Python dictionary
data = json.loads(json_data)

# Convert the dictionary to a YAML string
yaml_data = yaml.dump(data, default_flow_style=False)

# Print the YAML string
print(yaml_data)
