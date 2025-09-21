# main.py
from network import RailwayNetwork
from scheduler_with_platforms import TrainScheduler

from visualize_schedule import plot_train_schedule


# Initialize network
network = RailwayNetwork()
stations = ["A", "B", "C", "D"]
for s in stations:
    network.add_station(s)
network.add_track("A", "B", 5, 60)
network.add_track("B", "C", 10, 80)
network.add_track("C", "D", 7, 70)
network.add_track("A", "C", 15, 50)

# Define trains
trains = [
    {"id": "T1", "type": "express", "priority": 1, "start": "A", "end": "D", "departure": 0, "speed": 60},
    {"id": "T2", "type": "passenger", "priority": 2, "start": "A", "end": "D", "departure": 1, "speed": 50},
    {"id": "T3", "type": "freight", "priority": 3, "start": "B", "end": "D", "departure": 3, "speed": 40},
]

# Schedule trains
scheduler = TrainScheduler(network, trains, platforms_per_station=2)
train_schedule = scheduler.schedule_trains()

# Print schedule
for train_id, schedule in train_schedule.items():
    print(f"\nSchedule for {train_id}:")
    for station, (arrival, platform) in schedule.items():
        print(f"{station}: arrival at {arrival} min, platform {platform}")


# Visualize the schedule
plot_train_schedule(train_schedule, trains)
# main.py (continued)
#from ascii_visualizer import ascii_train_timeline

#ascii_train_timeline(train_schedule, trains)
