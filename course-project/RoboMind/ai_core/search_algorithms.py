import collections
import heapq
from typing import Callable, List, Tuple, Optional

# Define the state type used in the project (e.g., (row, col))
StateType = Tuple[int, int]
# Define the signature for the successor function: 
# Returns list of (action, next_state, cost)
SuccessorFnType = Callable[[StateType], List[Tuple[str, StateType, float]]]

# --- Node/State Management Helper Class ---

class SearchNode:
    """
    Helper class to track information about a node during search.
    This structure ensures we don't mix up states (tuples) and nodes (objects).
    """
    def __init__(self, state: StateType, parent=None, action: Optional[str] = None, cost: float = 0.0, h_cost: float = 0.0):
        self.state = state
        self.parent = parent    # SearchNode
        self.action = action    # Action (str) taken to reach this node
        self.cost = cost        # Total path cost (g(n)) from start to this node
        self.h_cost = h_cost    # Heuristic cost (h(n)) for A*

    # Define the comparison for the Priority Queue (Heapq)
    def __lt__(self, other):
        """Compares nodes based on the estimated total cost f(n) = g(n) + h(n)."""
        # Note: For UCS, h_cost will be 0. For BFS, cost and h_cost are generally ignored 
        # but the queue/deque structure handles the prioritization.
        return (self.cost + self.h_cost) < (other.cost + other.h_cost)

# --- Core Utility Function ---

def reconstruct_path(goal_node: SearchNode) -> Tuple[List[str], float]:
    """
    Reconstructs the sequence of actions and returns the final cost.
    """
    path_actions = []
    current = goal_node
    final_cost = current.cost

    # Traverse back from the goal node to the start node using parent pointers
    while current.parent is not None:
        path_actions.append(current.action)
        current = current.parent
    
    path_actions.reverse()
    
    # We return the path actions and the final cost.
    return path_actions, final_cost 

# --- Search Algorithms ---

def bfs(start_state: StateType, is_goal_fn: Callable[[StateType], bool], get_successors_fn: SuccessorFnType) -> Tuple[Optional[List[str]], float, int]:
    """Breadth-First Search: Finds the shortest path in terms of steps."""
    
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
            # Goal reached. Reconstruct path.
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

    return None, 0.0, expanded_count # No path found

# ---

def ucs(start_state: StateType, is_goal_fn: Callable[[StateType], bool], get_successors_fn: SuccessorFnType) -> Tuple[Optional[List[str]], float, int]:
    """Uniform Cost Search: Finds the path with the lowest cumulative cost."""
    
    initial_node = SearchNode(start_state, cost=0.0)
    # Frontier: Priority Queue (heapq) of SearchNode objects, ordered by g(n)
    frontier = [initial_node] 
    
    # min_cost_g: Maps state (tuple) -> min cost (float) found so far
    min_cost_g = {start_state: 0.0}
    expanded_count = 0
    
    while frontier:
        # Pop the node with the lowest cost (g(n))
        current_node = heapq.heappop(frontier)
        current_state = current_node.state
        expanded_count += 1

        # Check for redundant/sub-optimal paths from the queue
        if current_node.cost > min_cost_g[current_state]:
            continue

        if is_goal_fn(current_state):
            path, cost = reconstruct_path(current_node)
            return path, cost, expanded_count

        # Expand the current node
        for action, next_state, step_cost in get_successors_fn(current_state):
            new_cost_g = current_node.cost + step_cost
            
            # Check if this new path is better than any previously found path
            if next_state not in min_cost_g or new_cost_g < min_cost_g[next_state]:
                
                min_cost_g[next_state] = new_cost_g
                
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=new_cost_g
                )
                
                heapq.heappush(frontier, successor_node)
                
    return None, 0.0, expanded_count # No path found

# ---

def astar(start_state: StateType, is_goal_fn: Callable[[StateType], bool], get_successors_fn: SuccessorFnType, heuristic_fn: Callable[[StateType], float]) -> Tuple[Optional[List[str]], float, int]:
    """A* Search: Finds the lowest-cost path using a heuristic f(n) = g(n) + h(n)."""
    
    # Initialize costs and node
    initial_h = heuristic_fn(start_state)
    initial_node = SearchNode(start_state, cost=0.0, h_cost=initial_h)
    
    # Frontier: Priority Queue (heapq) of SearchNode objects, prioritized by f(n)
    frontier = [initial_node] 
    
    # min_cost_g: Maps state (tuple) -> min g-cost (float) found so far
    min_cost_g = {start_state: 0.0}
    expanded_count = 0

    while frontier:
        # Pop the element with the lowest f(n) cost
        current_node = heapq.heappop(frontier)
        current_state = current_node.state
        expanded_count += 1

        # Check for redundant/sub-optimal paths
        if current_node.cost > min_cost_g[current_state]:
            continue

        if is_goal_fn(current_state):
            path, cost = reconstruct_path(current_node)
            return path, cost, expanded_count

        # Expand the current node
        for action, next_state, step_cost in get_successors_fn(current_state):
            new_cost_g = current_node.cost + step_cost
            
            # Check if this new path is better than any previously found path
            if next_state not in min_cost_g or new_cost_g < min_cost_g[next_state]:
                
                min_cost_g[next_state] = new_cost_g
                
                # Calculate the h-cost (Heuristic call: this was a likely source of previous errors)
                new_cost_h = heuristic_fn(next_state)
                
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=new_cost_g, # g(n)
                    h_cost=new_cost_h # h(n)
                )
                
                heapq.heappush(frontier, successor_node)
                
    return None, 0.0, expanded_count # No path found
