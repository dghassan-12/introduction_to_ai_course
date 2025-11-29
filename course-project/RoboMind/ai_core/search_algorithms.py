from environment import GridWorld
from typing import List, Tuple, Optional
import heapq

# -----------------------------

# Breadth-First Search (BFS)

# -----------------------------

def bfs(env: GridWorld, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int,int]]], float, int]:
from collections import deque

```
frontier = deque([start])
came_from = {start: None}
expanded = 0

while frontier:
    current = frontier.popleft()
    expanded += 1
    
    if env.is_goal(current):
        # reconstruct path
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        cost = len(path) - 1
        return path, cost, expanded
    
    for neighbor in env.get_neighbors(current):
        if neighbor not in came_from:
            came_from[neighbor] = current
            frontier.append(neighbor)

return None, 0, expanded
```

# -----------------------------

# Uniform Cost Search (UCS)

# -----------------------------

def ucs(env: GridWorld, start: Tuple[int,int], goal: Tuple[int,int]) -> Tuple[Optional[List[Tuple[int,int]]], float, int]:
frontier = []
heapq.heappush(frontier, (0, start))
came_from = {start: None}
cost_so_far = {start: 0}
expanded = 0

```
while frontier:
    current_cost, current = heapq.heappop(frontier)
    expanded += 1
    
    if env.is_goal(current):
        # reconstruct path
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path, current_cost, expanded
    
    for neighbor in env.get_neighbors(current):
        new_cost = current_cost + env.get_cost(current, neighbor)
        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current
            heapq.heappush(frontier, (new_cost, neighbor))

return None, 0, expanded
```

# -----------------------------

# A* Search

# -----------------------------

def astar(env: GridWorld, start: Tuple[int,int], goal: Tuple[int,int], heuristic: str='manhattan') -> Tuple[Optional[List[Tuple[int,int]]], float, int]:
# Select heuristic function
if heuristic == 'manhattan':
h_func = env.manhattan_distance
elif heuristic == 'euclidean':
h_func = env.euclidean_distance
else:
raise ValueError(f"Unknown heuristic: {heuristic}")

```
frontier = []
heapq.heappush(frontier, (0 + h_func(start, goal), 0, start))
came_from = {start: None}
cost_so_far = {start: 0}
expanded = 0

while frontier:
    _, current_cost, current = heapq.heappop(frontier)
    expanded += 1
    
    if env.is_goal(current):
        # reconstruct path
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path, current_cost, expanded
    
    for neighbor in env.get_neighbors(current):
        new_cost = cost_so_far[current] + env.get_cost(current, neighbor)
        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current
            priority = new_cost + h_func(neighbor, goal)
            heapq.heappush(frontier, (priority, new_cost, neighbor))

return None, 0, expanded
```
