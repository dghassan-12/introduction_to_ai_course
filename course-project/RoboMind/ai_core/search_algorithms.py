import heapq
from collections import deque

class Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, u, v, cost=1):
        if u not in self.edges:
            self.edges[u] = []
        self.edges[u].append((v, cost))

    def neighbors(self, node):
        return self.edges.get(node, [])


# ---------------------------------------------------------
# Depth-First Search (DFS)
# ---------------------------------------------------------
def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        
        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph.neighbors(node):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None


# ---------------------------------------------------------
# Breadth-First Search (BFS)
# ---------------------------------------------------------
def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph.neighbors(node):
                queue.append((neighbor, path + [neighbor]))

    return None


# ---------------------------------------------------------
# Uniform Cost Search (UCS)
# ---------------------------------------------------------
def ucs(graph, start, goal):
    pq = [(0, start, [start])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == goal:
            return path, cost

        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in graph.neighbors(node):
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

    return None


# ---------------------------------------------------------
# A* Search
# ---------------------------------------------------------
def astar(graph, start, goal, heuristic):
    pq = [(heuristic(start), 0, start, [start])]
    visited = set()

    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node == goal:
            return path, g

        if node not in visited:
            visited.add(node)
            for neighbor, cost in graph.neighbors(node):
                new_cost = g + cost
                f_value = new_cost + heuristic(neighbor)
                heapq.heappush(pq, (f_value, new_cost, neighbor, path + [neighbor]))

    return None


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    g = Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("A", "C", 4)
    g.add_edge("B", "D", 2)
    g.add_edge("C", "D", 1)
    g.add_edge("D", "E", 3)

    print("DFS:", dfs(g, "A", "E"))
    print("BFS:", bfs(g, "A", "E"))
    print("UCS:", ucs(g, "A", "E"))
    
    heuristic = lambda n: {"A":4, "B":3, "C":2, "D":1, "E":0}[n]
    print("A*:", astar(g, "A", "E", heuristic))
