import sys
from io import StringIO


# def solve(input_str):
#     sys.stdin = StringIO(input_str)
#     for __ in range(int(input())):
#         n, k = map(int, sys.stdin.readline().split())
#         lists = map(int, sys.stdin.readline().split())
#         dp = [0] * (k + 1)
#         dp[0] = 1
#         for i in lists:
#             for j in range(k - i, -1, -1):
#                 if dp[k]:
#                     break
#                 if dp[j]:
#                     dp[j + i] = 1
#         print(dp[k])


import sys

input = sys.stdin.read
data = input().splitlines()

T = int(data[0])
results = []

index = 1
for _ in range(T):
    N, K = map(int, data[index].split())
    if N == 0:
        results.append(0)
        index += 1
        continue
    arr = list(map(int, data[index + 1].split()))

    dp = [[False] * (K + 1) for _ in range(N + 1)]
    dp[0][0] = True  # 0 sum is always possible with 0 elements

    for i in range(1, N + 1):
        dp[i][0] = True  # 0 sum is possible with any number of elements
        for j in range(1, K + 1):
            if arr[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - arr[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]

    results.append(1 if dp[N][K] else 0)
    index += 2

print('\n'.join(map(str, results)))


