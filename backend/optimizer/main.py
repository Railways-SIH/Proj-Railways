# main.py
from network import RailwayNetwork
from scheduler_with_platforms import TrainScheduler
from visualize_schedule import plot_train_schedule

# Station → Frontend Section mapping
station_to_section = {
    "A": "2R",
    "B": "3L",
    "C": "6L",
    "D": "9L"
}

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

# 🔹 Convert train + schedule to section-based IDs
response = {
    "trains": [
        {
            "id": t["id"],
            "name": t["id"],                 # You can replace with real train name
            "number": f"00{idx+1}",          # Simple numbering
            "start": station_to_section[t["start"]],
            "end": station_to_section[t["end"]],
            "speed": t["speed"]
        }
        for idx, t in enumerate(trains)
    ],
    "schedule": {
        tid: {
            station_to_section[s]: (arr, pf)
            for s, (arr, pf) in sch.items()
        }
        for tid, sch in train_schedule.items()
    }
}

# Debug print in console
for train_id, schedule in response["schedule"].items():
    print(f"\nSchedule for {train_id}:")
    for station, (arrival, platform) in schedule.items():
        print(f"{station}: arrival at {arrival} min, platform {platform}")

# Optional: Matplotlib visualization (can remove if running in API mode)
# plot_train_schedule(train_schedule, trains)

# If running as API with FastAPI/Flask, return `response` instead of printing
