# pathfinder.py
import heapq

class PathFinder:
    def __init__(self, network):
        self.network = network  # RailwayNetwork object

    def shortest_path(self, start, end):
        """
        Returns the shortest path from start to end based on track length.
        Does not currently check for block occupancy; can be extended.
        """
        graph = self.network.graph
        queue = [(0, start, [])]  # (cost_so_far, current_node, path_so_far)
        visited = set()

        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if node == end:
                return path, cost  # path and total length

            for neighbor, attr in graph.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + attr["length"], neighbor, path))
        return None, float('inf')  # no path found
