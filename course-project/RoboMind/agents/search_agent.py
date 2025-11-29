"""
Search Agent - RoboMind Project
SE444 - Artificial Intelligence Course Project
"""

from environment import GridWorld
from typing import Tuple, List, Optional
# Ensure the correct imports for your algorithms
from ai_core.search_algorithms import bfs, ucs, astar 


class SearchAgent:
    """
    An agent that uses search algorithms to navigate the grid world.
    """
    
    def __init__(self, environment: GridWorld):
        """
        Initialize the search agent.
        """
        self.env = environment
        self.path = [] # Stores list of actions (e.g., 'up', 'right')
        self.current_pos = environment.start
        
    # --- START OF MODIFIED METHOD ---
    def search(self, algorithm='bfs', heuristic='manhattan') -> Tuple[Optional[List], float, int]:
        """
        Find a path from start to goal using the specified algorithm.
        
        This method defines and passes the required callable functions 
        (goal test, successors, heuristic) to the search algorithms.
        """
        print(f"\n🔍 Running {algorithm.upper()} search...")
        print(f"   Start: {self.env.start}")
        print(f"   Goal: {self.env.goal}")
        
        # Define necessary arguments for the search algorithms:
        start_state = self.env.start 
        is_goal_fn = self.env.is_goal        # The environment method (callable)
        get_successors_fn = self.env.get_successors # The environment method (callable)
        
        path_actions = None
        cost = 0.0
        expanded = 0
        
        # Call the appropriate search algorithm
        if algorithm == 'bfs':
            # Pass the start state (tuple) and the two required functions (callable)
            path_actions, cost, expanded = bfs(
                start_state, is_goal_fn, get_successors_fn
            )
        elif algorithm == 'ucs':
            # Pass the start state (tuple) and the two required functions (callable)
            path_actions, cost, expanded = ucs(
                start_state, is_goal_fn, get_successors_fn
            )
        elif algorithm == 'astar':
            
            # Define the heuristic function (callable) needed for A*
            if heuristic == 'manhattan':
                # Lambda creates a function that takes *one* argument (state) and returns the distance to the fixed goal
                heuristic_fn = lambda state: self.env.manhattan_distance(state, self.env.goal)
            elif heuristic == 'euclidean':
                heuristic_fn = lambda state: self.env.euclidean_distance(state, self.env.goal)
            else:
                raise ValueError(f"Unknown heuristic: {heuristic}")
                
            # Pass the start state, the two required functions, and the heuristic function
            path_actions, cost, expanded = astar(
                start_state, is_goal_fn, get_successors_fn, heuristic_fn
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        self.path = path_actions
        
        return path_actions, cost, expanded
    # --- END OF MODIFIED METHOD ---
    
    def move_along_path(self):
        """
        Move the agent along the computed path (for visualization).
        """
        if not self.path:
            print("No path to follow!")
            return
        
        print(f"\n🤖 Moving along path ({len(self.path)} steps)...")
        
        current_state = self.env.start
        
        for i, action in enumerate(self.path):
            # NOTE: This implementation assumes your environment has an 'execute_action' method
            # that takes the current state and action, and returns the resulting state/cost.
            
            # Placeholder for state transition logic
            new_pos, step_cost = self.env.execute_action(current_state, action)
            
            self.env.agent_pos = new_pos
            self.env.visited.add(new_pos)
            self.env.render()
            
            current_state = new_pos 
            
            # Check if reached goal
            if self.env.is_goal(new_pos):
                print(f"✓ Goal reached at step {i+1}!")
                break


# Example usage and testing
if __name__ == "__main__":
    # ... (rest of the __main__ block)
    # This block usually remains unchanged
    pass
