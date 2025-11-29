# search_algorithms.py
# Phase 1 Implementation: BFS, UCS, A*, Path Reconstruction

from heapq import heappush, heappop
from collections import deque

# ---------------------------------------------------------
# Helper: Reconstruct path from goal to start using parents
# ---------------------------------------------------------
def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# ---------------------------------------------------------
# Breadth‑First Search (BFS) – Unweighted shortest path
# ---------------------------------------------------------
def bfs(environment, start, goal):
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for n in environment.get_neighbors(current):
            if n not in came_from:
                came_from[n] = current
                queue.append(n)

    return None

# ---------------------------------------------------------
# Uniform Cost Search (UCS) – Optimal for weighted costs
# ---------------------------------------------------------
def ucs(environment, start, goal):
    frontier = []
    heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for n in environment.get_neighbors(current):
            new_cost = current_cost + environment.get_cost(current, n)
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                came_from[n] = current
                heappush(frontier, (new_cost, n))

    return None

# ---------------------------------------------------------
# A* Search – Optimal + Heuristics
# ---------------------------------------------------------
def astar(environment, start, goal):
    frontier = []
    heappush(frontier, (0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        for n in environment.get_neighbors(current):
            new_cost = cost_so_far[current] + environment.get_cost(current, n)
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                came_from[n] = current
                priority = new_cost + environment.heuristic(n, goal)
                heappush(frontier, (priority, n))

    return None
