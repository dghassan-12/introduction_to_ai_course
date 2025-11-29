import collections
import heapq
from typing import Callable, List, Tuple, Optional

# Define the state type used in the project (e.g., (row, col))
StateType = Tuple[int, int]
# Successor Fn Type: Returns list of (action: str, next_state: tuple, cost: float)
SuccessorFnType = Callable[[StateType], List[Tuple[str, StateType, float]]]

# --- Node/State Management Helper Class ---

class SearchNode:
    """Helper class to track path information for the search algorithms."""
    def __init__(self, state: StateType, parent=None, action: Optional[str] = None, cost: float = 0.0, h_cost: float = 0.0):
        self.state = state
        self.parent = parent    # SearchNode
        self.action = action    # Action (str)
        self.cost = cost        # g(n): Total path cost from start
        self.h_cost = h_cost    # h(n): Heuristic cost

    def __lt__(self, other):
        """Used by heapq to prioritize nodes by f(n) = g(n) + h(n)."""
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
    
    return path_actions, final_cost 

# --- Search Algorithms ---

def bfs(start_state: StateType, is_goal_fn: Callable[[StateType], bool], get_successors_fn: SuccessorFnType) -> Tuple[Optional[List[str]], float, int]:
    """Breadth-First Search: Finds the shortest path in terms of steps."""
    
    frontier = collections.deque([SearchNode(start_state)])
    explored = {start_state}
    expanded_count = 0

    while frontier:
        current_node = frontier.popleft()
        current_state = current_node.state
        expanded_count += 1

        if is_goal_fn(current_state):
            path, cost = reconstruct_path(current_node)
            return path, cost, expanded_count

        # This unpacks the successor tuple correctly: (str, tuple, float)
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
    """Uniform Cost Search: Finds the path with the lowest cumulative cost."""
    
    initial_node = SearchNode(start_state, cost=0.0)
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
    """A* Search: Finds the lowest-cost path using a heuristic f(n) = g(n) + h(n)."""
    
    initial_h = heuristic_fn(start_state)
    initial_node = SearchNode(start_state, cost=0.0, h_cost=initial_h)
    
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
                
                # CRITICAL: This line calls the heuristic function
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
