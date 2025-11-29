from heapq import heappush, heappop
from collections import deque

# ---------------------------------------------------------
# Breadth-First Search (BFS)
# ---------------------------------------------------------
def bfs_search(environment, start, goal):
    queue = deque([start])
    came_from = {start: None}
    visited = {start}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for neighbor in environment.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return reconstruct_path(came_from, start, goal)


# ---------------------------------------------------------
# Uniform Cost Search (UCS)
# ---------------------------------------------------------
def ucs_search(environment, start, goal):
    frontier = []
    heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heappop(frontier)

        if current == goal:
            break

        for neighbor in environment.get_neighbors(current):
            new_cost = current_cost + environment.get_cost(current, neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heappush(frontier, (new_cost, neighbor))

    return reconstruct_path(came_from, start, goal)


# ---------------------------------------------------------
# A* Search
# ---------------------------------------------------------
def a_star_search(environment, start, goal):
    frontier = []
    heappush(frontier, (0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heappop(frontier)

        if current == goal:
            break

        for neighbor in environment.get_neighbors(current):
            new_cost = cost_so_far[current] + environment.get_cost(current, neighbor)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost

                # Use Manhattan heuristic (environment provides this)
                h = environment.manhattan_distance(neighbor, goal)
                priority = new_cost + h

                came_from[neighbor] = current
                heappush(frontier, (priority, neighbor))

    return reconstruct_path(came_from, start, goal)


# ---------------------------------------------------------
# Path reconstruction helper
# ---------------------------------------------------------
def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return []  # no path found

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1]
