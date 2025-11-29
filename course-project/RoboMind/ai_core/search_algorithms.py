import collections
import heapq
from typing import Callable, List, Tuple, Optional

# Define the state type (e.g., (row, col) coordinates)
StateType = Tuple[int, int]
# Successor Function Type: Callable[[current_state], List of (action: str, next_state: tuple, cost: float)]
SuccessorFnType = Callable[[StateType], List[Tuple[str, StateType, float]]]

# --- Node/State Management Helper Class ---

class SearchNode:
    """Helper class to track path information for the search algorithms."""
    def __init__(self, state: StateType, parent=None, action: Optional[str] = None, cost: float = 0.0, h_cost: float = 0.0):
        self.state = state
        self.parent = parent    # Parent SearchNode
        self.action = action    # Action (str) taken to reach this node
        self.cost = cost        # g(n): Total path cost from start
        self.h_cost = h_cost    # h(n): Heuristic cost

    # Define the comparison for the Priority Queue (Heapq)
    def __lt__(self, other):
        """Compares nodes based on the estimated total cost f(n) = g(n) + h(n)."""
        return (self.cost + self.h_cost) < (other.cost + other.h_cost)

# --- Core Utility Function ---

def reconstruct_path(goal_node: SearchNode) -> Tuple[List[str], float]:
    """Reconstructs the path of actions and returns the final cost."""
    path_actions = []
    current = goal_node
    final_cost = current.cost

    while current.parent is not None:
        path_actions.append(current.action)
        current = current.parent
    
    path_actions.reverse()
    
    # Returns (List of actions, Total cost)
    return path_actions, final_cost 

# --- Search Algorithms ---

def bfs(start_state: StateType, is_goal_fn: Callable[[StateType], bool], get_successors_fn: SuccessorFnType) -> Tuple[Optional[List[str]], float, int]:
    """
    Breadth-First Search: Finds the shortest path in terms of steps.
    """
    # Frontier: FIFO Queue (deque) of SearchNode objects
    frontier = collections.deque([SearchNode(start_state)])
    
    # Explored: Set of visited states (tuples)
    explored = {start_state}
    expanded_count = 0

    while frontier:
        current_node = frontier.popleft()
        current_state = current_node.state
        expanded_count += 1

        if is_goal_fn(current_state):
            path, cost = reconstruct_path(current_node)
            return path, cost, expanded_count

        # Expand the current node
        for action, next_state, cost_to_move in get_successors_fn(current_state):
            if next_state not in explored:
                
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=current_node.cost + cost_to_move
                )
                explored.add(next_state)
                frontier.append(successor_node)

    return None, 0.0, expanded_count

# ---

def ucs(start_state: StateType, is_goal_fn: Callable[[StateType], bool], get_successors_fn: SuccessorFnType) -> Tuple[Optional[List[str]], float, int]:
    """
    Uniform Cost Search: Finds the path with the lowest cumulative cost.
    """
    
    initial_node = SearchNode(start_state, cost=0.0)
    # Frontier: Priority Queue (heapq) of SearchNode objects, ordered by g(n)
    frontier = [initial_node] 
    
    # min_cost_g: Tracks minimum cost found so far to reach a state
    min_cost_g = {start_state: 0.0}
    expanded_count = 0
    
    while frontier:
        current_node = heapq.heappop(frontier)
        current_state = current_node.state
        expanded_count += 1

        # Check for redundant/sub-optimal paths from the queue
        if current_node.cost > min_cost_g[current_state]:
            continue

        if is_goal_fn(current_state):
            path, cost = reconstruct_path(current_node)
            return path, cost, expanded_count

        for action, next_state, step_cost in get_successors_fn(current_state):
            new_cost_g = current_node.cost + step_cost
            
            if next_state not in min_cost_g or new_cost_g < min_cost_g[next_state]:
                
                min_cost_g[next_state] = new_cost_g
                
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=new_cost_g
                )
                
                heapq.heappush(frontier, successor_node)
                
    return None, 0.0, expanded_count

# ---

def astar(start_state: StateType, is_goal_fn: Callable[[StateType], bool], get_successors_fn: SuccessorFnType, heuristic_fn: Callable[[StateType], float]) -> Tuple[Optional[List[str]], float, int]:
    """
    A* Search: Finds the lowest-cost path using a heuristic f(n) = g(n) + h(n).
    
    """
    
    # Initialize costs and node
    initial_h = heuristic_fn(start_state)
    initial_node = SearchNode(start_state, cost=0.0, h_cost=initial_h)
    
    # Frontier: Priority Queue (heapq) of SearchNode objects, prioritized by f(n)
    frontier = [initial_node] 
    
    min_cost_g = {start_state: 0.0}
    expanded_count = 0

    while frontier:
        current_node = heapq.heappop(frontier)
        current_state = current_node.state
        expanded_count += 1

        if current_node.cost > min_cost_g[current_state]:
            continue

        if is_goal_fn(current_state):
            path, cost = reconstruct_path(current_node)
            return path, cost, expanded_count

        for action, next_state, step_cost in get_successors_fn(current_state):
            new_cost_g = current_node.cost + step_cost
            
            if next_state not in min_cost_g or new_cost_g < min_cost_g[next_state]:
                
                min_cost_g[next_state] = new_cost_g
                
                # Calculate the h-cost
                new_cost_h = heuristic_fn(next_state)
                
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=new_cost_g, 
                    h_cost=new_cost_h
                )
                
                heapq.heappush(frontier, successor_node)
                
    return None, 0.0, expanded_count
