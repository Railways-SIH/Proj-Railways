from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
import heapq
import uvicorn

app = FastAPI(title="Railway Traffic Control Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RailwayNetwork:
    def __init__(self):
        self.graph = {}
        self.block_status = {}

    def add_station(self, station_name: str):
        if station_name not in self.graph:
            self.graph[station_name] = {}

    def add_track(self, from_station: str, to_station: str, length: int, speed_limit: int):
        self.graph[from_station][to_station] = {"length": length, "speed_limit": speed_limit}
        self.graph[to_station][from_station] = {"length": length, "speed_limit": speed_limit}
        self.block_status[(from_station, to_station)] = None
        self.block_status[(to_station, from_station)] = None

    def is_block_free(self, from_station: str, to_station: str) -> bool:
        return self.block_status.get((from_station, to_station)) is None

    def occupy_block(self, from_station: str, to_station: str, train_id: str):
        self.block_status[(from_station, to_station)] = train_id
        self.block_status[(to_station, from_station)] = train_id

    def release_block(self, from_station: str, to_station: str):
        self.block_status[(from_station, to_station)] = None
        self.block_status[(to_station, from_station)] = None

    def get_neighbors(self, station: str) -> dict:
        return self.graph.get(station, {})

class PathFinder:
    def __init__(self, network: RailwayNetwork):
        self.network = network

    def shortest_path(self, start: str, end: str) -> Tuple[Optional[List[str]], float]:
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

class TrainScheduler:
    def __init__(self, network: RailwayNetwork, trains: List[dict], platforms_per_station: int = 2, headway_time: int = 2):
        self.network = network
        self.trains = trains
        self.platforms_per_station = platforms_per_station
        self.headway_time = headway_time
        self.pf = PathFinder(network)
        self.train_schedule = {}
        self.platform_status = {}
        
        for station in network.graph.keys():
            self.platform_status[station] = {i+1: 0 for i in range(platforms_per_station)}

    def schedule_trains(self) -> Dict[str, Dict[str, Tuple[int, int]]]:
        sorted_trains = sorted(self.trains, key=lambda t: t['priority'])

        for train in sorted_trains:
            path, _ = self.pf.shortest_path(train['start'], train['end'])
            if not path:
                print(f"No path found for {train['id']}")
                continue

            self.train_schedule[train['id']] = {}
            time = train['departure']

            # Schedule all intermediate sections for complete route visualization
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

    def assign_platform(self, station: str, arrival_time: int) -> Optional[int]:
        for p_no, available_at in self.platform_status[station].items():
            if available_at <= arrival_time:
                self.platform_status[station][p_no] = arrival_time
                return p_no
        return None

    def calculate_travel_time(self, from_station: str, to_station: str, train_speed: int) -> int:
        track = self.network.graph[from_station][to_station]
        length = track['length']
        speed_limit = track['speed_limit']
        speed = min(train_speed, speed_limit)
        return int((length / speed) * 60)

# Enhanced section mapping with proper sequence
SECTION_SEQUENCE = {
    "2R": {"next": "3L", "station": "A"},
    "3L": {"next": "4L", "station": "B"}, 
    "4L": {"next": "5L", "station": None},
    "5L": {"next": "6L", "station": None},
    "6L": {"next": "7L", "station": "C"},
    "7L": {"next": "8L", "station": None},
    "8L": {"next": "9L", "station": None},
    "9L": {"next": None, "station": "D"},
    # Branch lines
    "101L": {"next": "102L", "station": None},
    "102L": {"next": "103L", "station": None},
    "103L": {"next": "104L", "station": None},
    "104L": {"next": None, "station": None},
    "201L": {"next": "202L", "station": None},
    "202L": {"next": "203L", "station": None},
    "203L": {"next": "204L", "station": None},
    "204L": {"next": None, "station": None},
    "301Y": {"next": "302Y", "station": None},
    "302Y": {"next": "303Y", "station": None},
    "303Y": {"next": "304Y", "station": None},
    "304Y": {"next": None, "station": None}
}

STATION_TO_SECTION = {
    "A": "2R",
    "B": "3L", 
    "C": "6L",
    "D": "9L"
}

def generate_complete_route(start_section: str, end_section: str) -> List[str]:
    """Generate complete route including all intermediate sections"""
    route = [start_section]
    current = start_section
    
    # Main line route
    main_line_order = ["2R", "3L", "4L", "5L", "6L", "7L", "8L", "9L"]
    
    try:
        start_idx = main_line_order.index(start_section)
        end_idx = main_line_order.index(end_section)
        
        if start_idx < end_idx:
            route = main_line_order[start_idx:end_idx + 1]
        else:
            route = main_line_order[end_idx:start_idx + 1][::-1]
            
    except ValueError:
        # Handle branch lines or other routes
        while current and current != end_section:
            next_section = SECTION_SEQUENCE.get(current, {}).get("next")
            if next_section:
                route.append(next_section)
                current = next_section
            else:
                break
    
    return route

# Pydantic models
class TrainRequest(BaseModel):
    id: str
    type: str
    priority: int
    start: str
    end: str
    departure: int
    speed: int

class ScheduleRequest(BaseModel):
    trains: List[TrainRequest]

class TrainResponse(BaseModel):
    id: str
    name: str
    number: str
    start: str
    end: str
    speed: int
    route: List[str]

class ScheduleResponse(BaseModel):
    trains: List[TrainResponse]
    schedule: Dict[str, Dict[str, Tuple[int, int]]]

def initialize_network():
    network = RailwayNetwork()
    stations = ["A", "B", "C", "D"]
    for station in stations:
        network.add_station(station)
    
    # Add tracks with realistic distances
    network.add_track("A", "B", 5, 60)   # A->B: 5km, 60kmh
    network.add_track("B", "C", 10, 80)  # B->C: 10km, 80kmh
    network.add_track("C", "D", 7, 70)   # C->D: 7km, 70kmh
    network.add_track("A", "C", 15, 50)  # A->C: 15km, 50kmh (alternative)
    
    return network

@app.get("/")
async def root():
    return {"message": "Railway Traffic Control Backend API v1.0"}

@app.post("/schedule", response_model=ScheduleResponse)
async def create_schedule(request: ScheduleRequest):
    try:
        trains_dict = [train.dict() for train in request.trains]
        print(f"\nReceived Schedule Request for {len(trains_dict)} trains")
        # Create fresh network
        network = initialize_network()
        
        # Schedule trains
        scheduler = TrainScheduler(network, trains_dict, platforms_per_station=2)
        train_schedule = scheduler.schedule_trains()
        print("\nGenerated Schedule:")
        for train_id, stops in train_schedule.items():
            print(f"  {train_id}: {stops}")
        # Train names mapping
        train_names = {
            "T1": "Rajdhani Express",
            "T2": "Shatabdi Express", 
            "T3": "Duronto Express"
        }
        
        train_numbers = {
            "T1": "12301",
            "T2": "12002",
            "T3": "12259"
        }
        
        # Convert to frontend format with complete routes
        frontend_trains = []
        for train in trains_dict:
            start_section = STATION_TO_SECTION[train["start"]]
            end_section = STATION_TO_SECTION[train["end"]]
            complete_route = generate_complete_route(start_section, end_section)
            
            frontend_trains.append({
                "id": train["id"],
                "name": train_names.get(train["id"], train["id"]),
                "number": train_numbers.get(train["id"], "00000"),
                "start": start_section,
                "end": end_section,
                "speed": train["speed"],
                "route": complete_route
            })
        
        # Convert schedule to section-based with intermediate sections
        frontend_schedule = {}
        for train_id, schedule in train_schedule.items():
            frontend_schedule[train_id] = {}
            for station, (arrival, platform) in schedule.items():
                section_id = STATION_TO_SECTION[station]
                frontend_schedule[train_id][section_id] = (arrival, platform)
        
        return ScheduleResponse(
            trains=frontend_trains,
            schedule=frontend_schedule
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/default-schedule", response_model=ScheduleResponse)
async def get_default_schedule():
    """Get default schedule with realistic train data"""
    default_trains = [
        {"id": "T1", "type": "express", "priority": 1, "start": "A", "end": "D", "departure": 0, "speed": 80},
        {"id": "T2", "type": "passenger", "priority": 2, "start": "A", "end": "D", "departure": 2, "speed": 60},
        {"id": "T3", "type": "freight", "priority": 3, "start": "B", "end": "D", "departure": 1, "speed": 45},
    ]
    
    request = ScheduleRequest(trains=[TrainRequest(**train) for train in default_trains])
    return await create_schedule(request)

@app.get("/network-status")
async def get_network_status():
    """Get current network status"""
    network = initialize_network()
    
    return {
        "stations": list(network.graph.keys()),
        "sections": list(SECTION_SEQUENCE.keys()),
        "tracks": len(network.block_status) // 2,
        "section_mapping": STATION_TO_SECTION
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")