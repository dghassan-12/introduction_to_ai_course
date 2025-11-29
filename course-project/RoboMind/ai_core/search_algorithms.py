from ai_core import search_algorithms

class SearchAgent:
    def __init__(self, env):
        self.env = env

    def search(self, algorithm):
        # Reset environment counter
        self.env.expanded = 0

        # -------------------------
        # BFS
        # -------------------------
        if algorithm == "bfs":
            path = search_algorithms.bfs(self.env)

            if path is None:
                return None, None, self.env.expanded

            cost = len(path)          # BFS cost = number of steps
            return path, cost, self.env.expanded

        # -------------------------
        # UCS
        # -------------------------
        elif algorithm == "ucs":
            result = search_algorithms.ucs(self.env)

            if result is None:
                return None, None, self.env.expanded

            path, cost = result
            return path, cost, self.env.expanded

        # -------------------------
        # A*
        # -------------------------
        elif algorithm == "astar":
            result = search_algorithms.astar(self.env, heuristic="manhattan")

            if result is None:
                return None, None, self.env.expanded

            path, cost = result
            return path, cost, self.env.expanded

        # -------------------------
        # Unknown algorithm
        # -------------------------
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
