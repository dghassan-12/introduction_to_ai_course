from typing import Tuple, List, Optional
from collections import deque
import heapq
import math # Used for Euclidean distance in A* (sqrt)

# ============================================================================
# UTILITY FUNCTION
# ============================================================================

def reconstruct_path(parent: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstruct path from parent pointers.
    """
    path = []
    current = goal
    
    # Trace back from goal to start
    while current != start:
        if current not in parent or parent[current] is None and current != start:
            # Should not happen if a path was found and start had parent=None
            return []
        path.append(current)
        current = parent[current]
        
    # Add the start node
    path.append(start)
    
    # Path is currently goal -> start, so reverse it
    return path[::-1]

# ============================================================================
# 1. BREADTH-FIRST SEARCH (BFS)
# ============================================================================

def bfs(env, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List], float, int]:
    """
    Breadth-First Search - Find shortest path in terms of number of steps.
    """
    # Frontier: Queue (FIFO)
    queue = deque([start])
    # Explored set for visited nodes
    visited = {start}
    # Parent pointers
    parent = {start: None}
    
    expanded = 0
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            # Goal found! Reconstruct path.
            path = reconstruct_path(parent, start, goal)
            # Cost for BFS is the number of steps (path length - 1)
            cost = len(path) - 1 
            return path, float(cost), expanded

        expanded += 1
        
        for neighbor in env.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    # No path found
    return None, float('inf'), expanded

# ============================================================================
# 2. UNIFORM COST SEARCH (UCS)
# ============================================================================

def ucs(env, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List], float, int]:
    """
    Uniform Cost Search - Find path with lowest total cost. (Dijkstra's)
    """
    # Frontier: Priority Queue (cost, position)
    frontier = [(0, start)]  
    
    # Dictionary to store the lowest cost found so far to reach a node
    cost_so_far = {start: 0}
    parent = {start: None}
    
    expanded = 0
    
    while frontier:
        # Get node with the lowest current cost
        current_cost, current = heapq.heappop(frontier)
        
        # Check for optimality: If we pop the goal, we have the optimal path.
        if current == goal:
            path = reconstruct_path(parent, start, goal)
            return path, current_cost, expanded
            
        expanded += 1
        
        for neighbor in env.get_neighbors(current):
            # Calculate the cost to reach the neighbor through the current node
            step_cost = env.get_cost(current, neighbor)
            new_cost = current_cost + step_cost
            
            # Key check: relaxation (update cost if new path is cheaper)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = current
                # Push the neighbor to the frontier with its new total path cost
                heapq.heappush(frontier, (new_cost, neighbor))
    
    # No path found
    return None, float('inf'), expanded

# ============================================================================
# 3. A* SEARCH
# ============================================================================

def astar(env, start: Tuple[int, int], goal: Tuple[int, int], 
          heuristic='manhattan') -> Tuple[Optional[List], float, int]:
    """
    A* Search - Find optimal path using f(n) = g(n) + h(n).
    """
    # 1. Define Heuristic Function
    if heuristic == 'manhattan':
        h_func = lambda pos: env.manhattan_distance(pos, goal)
    elif heuristic == 'euclidean':
        h_func = lambda pos: env.euclidean_distance(pos, goal)
    else:
        # Fallback to a zero heuristic, making it UCS, but raise error as instructed
        raise ValueError(f"Unknown heuristic: {heuristic}")
        
    # g_score: Actual cost from start to current node (g(n))
    g_score = {start: 0}
    # f_score: Estimated total cost (f(n) = g(n) + h(n))
    f_score = {start: h_func(start)}
    
    # Frontier: Priority Queue (f_score, position)
    frontier = [(f_score[start], start)]  
    parent = {start: None}
    
    expanded = 0
    
    while frontier:
        # Pop the node with the lowest f_score
        current_f, current = heapq.heappop(frontier)
        
        if current == goal:
            path = reconstruct_path(parent, start, goal)
            # Return ACTUAL cost g(goal), not f(goal)
            return path, g_score[current], expanded
            
        expanded += 1
        
        for neighbor in env.get_neighbors(current):
            # tentative_g is the g(n) if we go through 'current'
            step_cost = env.get_cost(current, neighbor)
            tentative_g = g_score[current] + step_cost
            
            # Key check: relaxation (update cost if new path is cheaper)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # This is a better path. Record it.
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Update the f_score for the frontier
                f_score_new = tentative_g + h_func(neighbor)
                
                # Push the neighbor to the frontier with its new f_score
                heapq.heappush(frontier, (f_score_new, neighbor))
    
    # No path found
    return None, float('inf'), expanded
