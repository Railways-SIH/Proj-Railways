# phase1_demo_matplotlib.py

import heapq
import matplotlib.pyplot as plt

# -------------------------
# Railway Network
# -------------------------
class RailwayNetwork:
    def __init__(self):
        self.graph = {}
        self.block_status = {}

    def add_station(self, station_name):
        if station_name not in self.graph:
            self.graph[station_name] = {}

    def add_track(self, from_station, to_station, length, speed_limit):
        self.graph[from_station][to_station] = {"length": length, "speed_limit": speed_limit}
        self.graph[to_station][from_station] = {"length": length, "speed_limit": speed_limit}
        self.block_status[(from_station, to_station)] = None
        self.block_status[(to_station, from_station)] = None

    def is_block_free(self, from_station, to_station):
        return self.block_status[(from_station, to_station)] is None

    def occupy_block(self, from_station, to_station, train_id):
        self.block_status[(from_station, to_station)] = train_id
        self.block_status[(to_station, from_station)] = train_id

    def release_block(self, from_station, to_station):
        self.block_status[(from_station, to_station)] = None
        self.block_status[(to_station, from_station)] = None

# -------------------------
# PathFinder
# -------------------------
class PathFinder:
    def __init__(self, network):
        self.network = network

    def shortest_path(self, start, end):
        graph = self.network.graph
        queue = [(0, start, [])]
        visited = set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]
            if node == end:
                return path, cost
            for neighbor, attr in graph.get(node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + attr["length"], neighbor, path))
        return None, float('inf')

# -------------------------
# Train Scheduler with Platforms
# -------------------------
class TrainScheduler:
    def __init__(self, network, trains, platforms_per_station=2, headway_time=2):
        self.network = network
        self.trains = trains
        self.platforms_per_station = platforms_per_station
        self.headway_time = headway_time
        self.pf = PathFinder(network)
        self.train_schedule = {}
        self.platform_status = {}
        for station in network.graph.keys():
            self.platform_status[station] = {i+1: 0 for i in range(platforms_per_station)}

    def schedule_trains(self):
        sorted_trains = sorted(self.trains, key=lambda t: t['priority'])
        for train in sorted_trains:
            path, _ = self.pf.shortest_path(train['start'], train['end'])
            if not path:
                print(f"No path for {train['id']}")
                continue
            self.train_schedule[train['id']] = {}
            time = train['departure']
            for i in range(len(path)-1):
                from_station = path[i]
                to_station = path[i+1]
                while not self.network.is_block_free(from_station, to_station):
                    time += 1
                self.network.occupy_block(from_station, to_station, train['id'])
                travel_time = self.calculate_travel_time(from_station, to_station, train['speed'])
                arrival_time = time + travel_time
                platform_no = self.assign_platform(to_station, arrival_time)
                while not platform_no:
                    arrival_time += 1
                    platform_no = self.assign_platform(to_station, arrival_time)
                self.train_schedule[train['id']][to_station] = (arrival_time, platform_no)
                self.network.release_block(from_station, to_station)
                time = arrival_time + self.headway_time
        return self.train_schedule

    def assign_platform(self, station, arrival_time):
        for p_no, available_at in self.platform_status[station].items():
            if available_at <= arrival_time:
                self.platform_status[station][p_no] = arrival_time
                return p_no
        return None

    def calculate_travel_time(self, from_station, to_station, train_speed):
        track = self.network.graph[from_station][to_station]
        length = track['length']
        speed = min(train_speed, track['speed_limit'])
        return int((length / speed) * 60)

# -------------------------
# Matplotlib Visualization
# -------------------------
def plot_train_schedule(train_schedule, trains):
    fig, ax = plt.subplots(figsize=(12, len(trains)*1.5))
    y_labels = []
    y_pos = []
    color_map = ["skyblue", "lightgreen", "salmon", "orange", "violet"]

    for idx, train in enumerate(trains):
        train_id = train["id"]
        y_labels.append(train_id)
        y_pos.append(idx*10)
        schedule = train_schedule.get(train_id, {})
        prev_time = train["departure"]
        prev_station = train["start"]
        color_idx = 0
        for station, (arrival, platform) in schedule.items():
            ax.barh(
                y=y_pos[idx],
                width=arrival - prev_time,
                left=prev_time,
                height=5,
                color=color_map[color_idx % len(color_map)],
                edgecolor="black",
            )
            ax.text(
                x=prev_time + (arrival - prev_time)/2,
                y=y_pos[idx]+1,
                s=f"P{platform}",
                ha="center",
                va="bottom",
                fontsize=9
            )
            prev_time = arrival
            prev_station = station
            color_idx += 1

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Train Movement Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# -------------------------
# Demo Run
# -------------------------
if __name__ == "__main__":
    network = RailwayNetwork()
    stations = ["A", "B", "C", "D", "E"]
    for s in stations:
        network.add_station(s)
    network.add_track("A", "B", 5, 60)
    network.add_track("B", "C", 8, 70)
    network.add_track("C", "D", 6, 60)
    network.add_track("D", "E", 10, 80)
    network.add_track("A", "C", 12, 50)
    network.add_track("B", "D", 15, 50)

    trains = [
        {"id": "T1", "type": "express", "priority": 1, "start": "A", "end": "E", "departure": 0, "speed": 60},
        {"id": "T2", "type": "passenger", "priority": 2, "start": "A", "end": "D", "departure": 1, "speed": 50},
        {"id": "T3", "type": "freight", "priority": 3, "start": "B", "end": "E", "departure": 2, "speed": 40},
        {"id": "T4", "type": "express", "priority": 1, "start": "C", "end": "E", "departure": 3, "speed": 60},
    ]

    scheduler = TrainScheduler(network, trains, platforms_per_station=2)
    train_schedule = scheduler.schedule_trains()

    for train_id, schedule in train_schedule.items():
        print(f"\nSchedule for {train_id}:")
        for station, (arrival, platform) in schedule.items():
            print(f"{station}: arrival at {arrival} min, platform {platform}")

    plot_train_schedule(train_schedule, trains)
