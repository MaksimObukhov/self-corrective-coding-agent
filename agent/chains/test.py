def solve(input_str: str):
    import sys
    from collections import deque, defaultdict

    input_data = input_str.strip().split('\n')
    n, m = map(int, input_data[0].split())

    # Adjacency list for roads
    roads = defaultdict(lambda: [[], []])  # roads[town] = [pedestrian_roads, bike_roads]

    for i in range(1, m + 1):
        vi, ui, ti = map(int, input_data[i].split())
        roads[vi][ti].append(ui)

    # BFS setup
    queue = deque([(1, 0)])  # (current_town, current_road_type)
    max_length = defaultdict(lambda: [0, 0])  # max_length[town] = [max_length_pedestrian, max_length_bike]
    max_length[1][0] = 1  # Start from town 1 with a pedestrian road

    # BFS loop
    while queue:
        current_town, current_road_type = queue.popleft()
        next_road_type = 1 - current_road_type  # Flip road type

        for next_town in roads[current_town][current_road_type]:
            if max_length[next_town][next_road_type] < max_length[current_town][current_road_type] + 1:
                max_length[next_town][next_road_type] = max_length[current_town][current_road_type] + 1
                if max_length[next_town][next_road_type] > 10**18:
                    print(-1)
                    return
                queue.append((next_town, next_road_type))

    # Find the maximum length
    result = max(max_length[town][road_type] for town in range(1, n + 1) for road_type in [0, 1])
    print(result - 1)  # Subtract 1 because we started counting from 1


if __name__ == '__main__':
    solve('1 2\n1 1 0\n1 1 1\n')