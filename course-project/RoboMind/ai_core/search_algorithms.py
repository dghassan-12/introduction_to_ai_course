import heapq
from collections import deque

# ---------------------------------------------------------
# Breadth-First Search (BFS)
# ---------------------------------------------------------
def bfs(env):
    start = env.start
    goal = env.goal

    frontier = deque([start])
    visited = set([start])
    parent = {start: None}

    while frontier:
        node = frontier.popleft()

        if node == goal:
            break

        for neighbor in env.get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                frontier.append(neighbor)
                env.expanded += 1

    if goal not in parent:
        return None

    # Reconstruct path
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()

    return path


# ---------------------------------------------------------
# Uniform Cost Search (UCS)
# ---------------------------------------------------------
def ucs(env):
    start = env.start
    goal = env.goal

    pq = [(0, start)]
    parent = {start: None}
    cost_so_far = {start: 0}
    visited = set()

    while pq:
        cost, node = heapq.heappop(pq)

        if node == goal:
            break

        if node in visited:
            continue
        visited.add(node)

        for neighbor in env.get_neighbors(node):
            new_cost = cost + env.get_cost(node, neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(pq, (new_cost, neighbor))
                env.expanded += 1

    if goal not in parent:
        return None

    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()

    return path, cost_so_far[goal]


# ---------------------------------------------------------
# A* Search
# ---------------------------------------------------------
def astar(env, heuristic="manhattan"):
    start = env.start
    goal = env.goal

    if heuristic == "manhattan":
        h = lambda n: env.manhattan_distance(n, goal)
    elif heuristic == "euclidean":
        h = lambda n: env.euclidean_distance(n, goal)
    else:
        raise ValueError("Unknown heuristic")

    pq = [(h(start), 0, start)]
    parent = {start: None}
    cost_so_far = {start: 0}
    visited = set()

    while pq:
        f, g, node = heapq.heappop(pq)

        if node == goal:
            break

        if node in visited:
            continue
        visited.add(node)

        for neighbor in env.get_neighbors(node):
            new_cost = g + env.get_cost(node, neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = node
                f_value = new_cost + h(neighbor)
                heapq.heappush(pq, (f_value, new_cost, neighbor))
                env.expanded += 1

    if goal not in parent:
        return None

    # Reconstruct path
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    path.reverse()

    return path, cost_so_far[goal]
