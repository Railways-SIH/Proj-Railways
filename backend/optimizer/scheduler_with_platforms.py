# scheduler_with_platforms.py
from pathfinder import PathFinder

class TrainScheduler:
    def __init__(self, network, trains, platforms_per_station=2, headway_time=2):
        self.network = network
        self.trains = trains
        self.platforms_per_station = platforms_per_station
        self.headway_time = headway_time
        self.pf = PathFinder(network)
        self.train_schedule = {}  # train_id -> {station: (arrival_time, platform_no)}
        self.platform_status = {}  # station -> {platform_no: available_at_time}
        for station in network.graph.keys():
            self.platform_status[station] = {i+1: 0 for i in range(platforms_per_station)}

    def schedule_trains(self):
        # Sort trains by priority
        sorted_trains = sorted(self.trains, key=lambda t: t['priority'])

        for train in sorted_trains:
            path, _ = self.pf.shortest_path(train['start'], train['end'])
            if not path:
                print(f"No path found for {train['id']}")
                continue

            self.train_schedule[train['id']] = {}
            time = train['departure']

            for i in range(len(path)-1):
                from_station = path[i]
                to_station = path[i+1]

                # Wait if block is occupied
                while not self.network.is_block_free(from_station, to_station):
                    time += 1  # simple 1-minute wait

                # Occupy the block
                self.network.occupy_block(from_station, to_station, train['id'])

                # Calculate travel time
                travel_time = self.calculate_travel_time(from_station, to_station, train['speed'])
                arrival_time = time + travel_time

                # Assign platform at destination
                platform_no = self.assign_platform(to_station, arrival_time)
                if not platform_no:
                    # Delay train until platform is free
                    while not platform_no:
                        arrival_time += 1
                        platform_no = self.assign_platform(to_station, arrival_time)

                # Record arrival + platform
                self.train_schedule[train['id']][to_station] = (arrival_time, platform_no)

                # Release block after travel
                self.network.release_block(from_station, to_station)

                # Move time forward + headway
                time = arrival_time + self.headway_time

        return self.train_schedule

    def assign_platform(self, station, arrival_time):
        """Return first available platform number at station for given arrival time"""
        for p_no, available_at in self.platform_status[station].items():
            if available_at <= arrival_time:
                self.platform_status[station][p_no] = arrival_time  # update occupancy
                return p_no
        return None  # no platform available

    def calculate_travel_time(self, from_station, to_station, train_speed):
        track = self.network.graph[from_station][to_station]
        length = track['length']
        speed_limit = track['speed_limit']
        speed = min(train_speed, speed_limit)
        return int((length / speed) * 60)  # travel time in minutes
