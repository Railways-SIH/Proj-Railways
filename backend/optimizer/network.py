# network.py
class RailwayNetwork:
    def __init__(self):
        # Graph representation: node -> neighbors with edge attributes
        self.graph = {}
        # Block occupancy: (from_node, to_node) -> train_id or None
        self.block_status = {}

    def add_station(self, station_name):
        if station_name not in self.graph:
            self.graph[station_name] = {}

    def add_track(self, from_station, to_station, length, speed_limit):
        # Add bidirectional track
        self.graph[from_station][to_station] = {"length": length, "speed_limit": speed_limit}
        self.graph[to_station][from_station] = {"length": length, "speed_limit": speed_limit}
        # Initialize block as free
        self.block_status[(from_station, to_station)] = None
        self.block_status[(to_station, from_station)] = None

    def is_block_free(self, from_station, to_station):
        return self.block_status.get((from_station, to_station)) is None

    def occupy_block(self, from_station, to_station, train_id):
        self.block_status[(from_station, to_station)] = train_id
        self.block_status[(to_station, from_station)] = train_id

    def release_block(self, from_station, to_station):
        self.block_status[(from_station, to_station)] = None
        self.block_status[(to_station, from_station)] = None

    def get_neighbors(self, station):
        return self.graph.get(station, {})
