# scheduler.py
from pathfinder import PathFinder

class TrainScheduler:
    def __init__(self, network, trains, headway_time=2):
        self.network = network
        self.trains = trains  # list of train dicts
        self.headway_time = headway_time  # minimum time gap between trains on the same block
        self.pf = PathFinder(network)
        self.train_schedule = {}  # train_id -> {station: arrival_time}

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
            
            # Assign blocks sequentially
            for i in range(len(path)-1):
                from_station = path[i]
                to_station = path[i+1]
                
                # Wait if block is occupied
                while not self.network.is_block_free(from_station, to_station):
                    time += 1  # simple 1-minute wait

                # Occupy the block
                self.network.occupy_block(from_station, to_station, train['id'])

                # Record arrival at next station
                travel_time = self.calculate_travel_time(from_station, to_station, train['speed'])
                arrival_time = time + travel_time
                self.train_schedule[train['id']][to_station] = arrival_time

                # Release the block after travel
                self.network.release_block(from_station, to_station)

                # Move time forward + headway
                time = arrival_time + self.headway_time

        return self.train_schedule

    def calculate_travel_time(self, from_station, to_station, train_speed):
        track = self.network.graph[from_station][to_station]
        length = track['length']
        speed_limit = track['speed_limit']
        speed = min(train_speed, speed_limit)
        # travel time in minutes (assuming speed in km/h and length in km)
        return int((length / speed) * 60)
