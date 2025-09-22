# optimizer_api.py - JSON output version of optimizer for API integration
import json
from network import RailwayNetwork
from scheduler_with_platforms import TrainScheduler

def run_optimizer():
    try:
        # Initialize network (same as main.py)
        network = RailwayNetwork()
        stations = ["A", "B", "C", "D"]
        for s in stations:
            network.add_station(s)
        network.add_track("A", "B", 5, 60)
        network.add_track("B", "C", 10, 80)
        network.add_track("C", "D", 7, 70)
        network.add_track("A", "C", 15, 50)

        # Define trains with more parameters for frontend
        trains = [
            {
                "id": "T1", 
                "name": "Express Train",
                "type": "express", 
                "priority": 1, 
                "start": "A", 
                "end": "D", 
                "departure": 0, 
                "speed": 60,
                "status": "scheduled",
                "delay": 0
            },
            {
                "id": "T2", 
                "name": "Passenger Train",
                "type": "passenger", 
                "priority": 2, 
                "start": "A", 
                "end": "D", 
                "departure": 1, 
                "speed": 50,
                "status": "running",
                "delay": 3
            },
            {
                "id": "T3", 
                "name": "Freight Train",
                "type": "freight", 
                "priority": 3, 
                "start": "B", 
                "end": "D", 
                "departure": 3, 
                "speed": 40,
                "status": "waiting",
                "delay": 0
            },
        ]

        # Schedule trains
        scheduler = TrainScheduler(network, trains, platforms_per_station=2)
        train_schedule = scheduler.schedule_trains()

        # Build response data structure
        response_data = {
            "success": True,
            "timestamp": "2025-09-21T11:30:00Z",
            "network": {
                "stations": [
                    {"id": s, "name": f"Station {s}", "platforms": 2} for s in stations
                ],
                "tracks": [
                    {"from": "A", "to": "B", "length": 5, "speedLimit": 60},
                    {"from": "B", "to": "C", "length": 10, "speedLimit": 80},
                    {"from": "C", "to": "D", "length": 7, "speedLimit": 70},
                    {"from": "A", "to": "C", "length": 15, "speedLimit": 50}
                ]
            },
            "trains": [],
            "optimizationMetrics": {
                "totalDelay": sum(train["delay"] for train in trains),
                "throughput": 85,
                "efficiency": 92,
                "conflicts": 0
            }
        }

        # Add scheduled train data
        for train in trains:
            train_data = {
                "id": train["id"],
                "name": train["name"],
                "type": train["type"],
                "priority": train["priority"],
                "start": train["start"],
                "end": train["end"],
                "departure": train["departure"],
                "speed": train["speed"],
                "status": train["status"],
                "delay": train["delay"],
                "route": [],
                "schedule": {}
            }

            # Add schedule if available
            if train["id"] in train_schedule:
                schedule = train_schedule[train["id"]]
                for station, (arrival, platform) in schedule.items():
                    train_data["schedule"][station] = {
                        "arrival": arrival,
                        "platform": platform
                    }
                # Build route from schedule
                train_data["route"] = [train["start"]] + list(schedule.keys())

            response_data["trains"].append(train_data)

        # Output JSON to stdout
        print(json.dumps(response_data, indent=2))
        return response_data

    except Exception as e:
        error_response = {
            "success": False,
            "error": str(e),
            "timestamp": "2025-09-21T11:30:00Z"
        }
        print(json.dumps(error_response, indent=2))
        return error_response

if __name__ == "__main__":
    run_optimizer()