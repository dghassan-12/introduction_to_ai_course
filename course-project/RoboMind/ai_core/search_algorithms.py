import collections
import heapq

# --- Node/State Management Helper Class (Optional but recommended) ---
# Assuming your states are hashable (e.g., tuples of coordinates), 
# we need a way to store the path details: parent, action, and cost.

# The 'start_state' will be a single state object (e.g., a tuple (x, y)).
# The 'get_successors' function returns a list of (action, next_state, cost).

class SearchNode:
    """Helper class to track information about a node during search."""
    def __init__(self, state, parent=None, action=None, cost=0.0):
        self.state = state
        self.parent = parent    # The node from which this node was reached
        self.action = action    # The action taken to reach this node
        self.cost = cost        # Total path cost (g(n)) from start to this node

    def __lt__(self, other):
        """Needed for heapq comparison. Compares nodes based on cost."""
        return self.cost < other.cost

# --- Core Utility Function ---

def reconstruct_path(goal_node):
    """
    Reconstructs the sequence of actions from the start state to the goal state.

    Args:
        goal_node (SearchNode): The final node reached by the search algorithm.

    Returns:
        list: A list of actions (e.g., ['up', 'right', ...])
    """
    path = []
    current = goal_node
    # Traverse back from the goal node to the start node using parent pointers
    while current.parent is not None:
        path.append(current.action)
        current = current.parent
    
    # The actions are in reverse order, so we need to reverse the list
    path.reverse()
    return path

# --- Search Algorithms ---

def bfs(start_state, is_goal_fn, get_successors_fn):
    """
    Breadth-First Search: Finds the shortest path in terms of steps.
    
    Args:
        start_state (object): The initial state of the agent.
        is_goal_fn (function): Takes a state and returns True if it's the goal.
        get_successors_fn (function): Takes a state and returns 
                                      [(action, next_state, cost), ...].

    Returns:
        list: A list of actions that form the path, or None if no path found.
    """
    # Use a deque as the FIFO Queue for the frontier
    frontier = collections.deque([SearchNode(start_state)])
    
    # Use a set for the explored/visited states
    explored = {start_state}

    # BFS guarantees shortest path, so we don't need to check for better paths
    # to nodes already in the explored set.

    while frontier:
        current_node = frontier.popleft()
        current_state = current_node.state

        if is_goal_fn(current_state):
            # Goal reached. Reconstruct and return the path.
            return reconstruct_path(current_node)

        # Expand the current node
        for action, next_state, cost in get_successors_fn(current_state):
            if next_state not in explored:
                # Create a new node for the successor
                new_cost = current_node.cost + cost
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=new_cost # Note: cost isn't strictly needed for standard BFS but kept for uniform structure
                )
                explored.add(next_state)
                frontier.append(successor_node)

    return None # No path found

def ucs(start_state, is_goal_fn, get_successors_fn):
    """
    Uniform Cost Search: Finds the path with the lowest cumulative cost.

    Args:
        start_state (object): The initial state of the agent.
        is_goal_fn (function): Takes a state and returns True if it's the goal.
        get_successors_fn (function): Takes a state and returns 
                                      [(action, next_state, cost), ...].

    Returns:
        list: A list of actions that form the path, or None if no path found.
    """
    # Priority Queue for the frontier: stores (cost, node) tuples
    # Cost is the priority key, so the node with the lowest cost is popped first.
    # The SearchNode class's __lt__ handles the cost comparison implicitly
    initial_node = SearchNode(start_state, cost=0.0)
    # Store in the heap as (cost, state_tuple, node_object)
    frontier = [(0.0, start_state, initial_node)] 
    heapq.heapify(frontier)
    
    # Track the minimum cost found so far to reach a state.
    # Maps state -> min_cost_g
    min_cost_g = {start_state: 0.0}
    
    # Track parent pointers/path reconstruction directly
    came_from = {start_state: initial_node}

    while frontier:
        # Pop the element with the lowest cost
        cost_g, current_state, current_node = heapq.heappop(frontier)

        # Check if we have already found a cheaper path to this state
        if cost_g > min_cost_g[current_state]:
            continue

        if is_goal_fn(current_state):
            return reconstruct_path(current_node)

        # Expand the current node
        for action, next_state, step_cost in get_successors_fn(current_state):
            new_cost_g = current_node.cost + step_cost
            
            # Check if this new path is better than any previously found path
            if next_state not in min_cost_g or new_cost_g < min_cost_g[next_state]:
                
                # Update records for the new better path
                min_cost_g[next_state] = new_cost_g
                
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=new_cost_g
                )
                came_from[next_state] = successor_node
                
                # Push the new path onto the priority queue
                heapq.heappush(frontier, (new_cost_g, next_state, successor_node))
                
    return None # No path found

def astar(start_state, is_goal_fn, get_successors_fn, heuristic_fn):
    """
    A* Search: Finds the lowest-cost path using a heuristic to guide the search.

    Args:
        start_state (object): The initial state of the agent.
        is_goal_fn (function): Takes a state and returns True if it's the goal.
        get_successors_fn (function): Takes a state and returns 
                                      [(action, next_state, cost), ...].
        heuristic_fn (function): Takes a state and returns a heuristic estimate (float).

    Returns:
        list: A list of actions that form the path, or None if no path found.
    """
    # A* priority is f(n) = g(n) + h(n)
    
    # Priority Queue for the frontier: stores (f_cost, state_tuple, node_object)
    # where f_cost = g_cost + h_cost
    initial_h = heuristic_fn(start_state)
    initial_f = 0.0 + initial_h
    initial_node = SearchNode(start_state, cost=0.0)
    
    # The heap is prioritized by the first element, initial_f
    frontier = [(initial_f, start_state, initial_node)] 
    heapq.heapify(frontier)
    
    # Track the minimum g-cost found so far to reach a state.
    # Maps state -> min_cost_g
    min_cost_g = {start_state: 0.0}
    
    # Track parent pointers/path reconstruction directly
    came_from = {start_state: initial_node}

    while frontier:
        # Pop the element with the lowest f(n) cost
        f_cost, current_state, current_node = heapq.heappop(frontier)

        # Check if we have already found a path with a lower g-cost
        # (Since we store the old f-cost in the heap, we must check if 
        # the g-cost is still the best known g-cost)
        if current_node.cost > min_cost_g[current_state]:
            continue

        if is_goal_fn(current_state):
            return reconstruct_path(current_node)

        # Expand the current node
        for action, next_state, step_cost in get_successors_fn(current_state):
            new_cost_g = current_node.cost + step_cost
            
            # Check if this new path is better than any previously found path
            if next_state not in min_cost_g or new_cost_g < min_cost_g[next_state]:
                
                # Update records for the new better path
                min_cost_g[next_state] = new_cost_g
                
                successor_node = SearchNode(
                    state=next_state, 
                    parent=current_node, 
                    action=action, 
                    cost=new_cost_g # This is g(n)
                )
                came_from[next_state] = successor_node
                
                # Calculate the f-cost: f(n) = g(n) + h(n)
                new_cost_h = heuristic_fn(next_state)
                new_cost_f = new_cost_g + new_cost_h
                
                # Push the new path onto the priority queue
                heapq.heappush(frontier, (new_cost_f, next_state, successor_node))
                
    return None # No path found
