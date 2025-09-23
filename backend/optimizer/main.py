from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Set, Tuple
import heapq
import uvicorn
import asyncio
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent Railway Control Backend", version="4.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TRACK_SECTIONS matching your updated frontend code
TRACK_SECTIONS = [
    {'id': 'STN_A', 'type': 'station', 'name': 'STA A', 'station': 'A', 'platforms': 3},
    {'id': 'BLOCK_A1', 'type': 'block', 'name': 'Block A1'},
    {'id': 'BLOCK_A2', 'type': 'block', 'name': 'Block A2'},
    {'id': 'STN_B', 'type': 'station', 'name': 'STA B', 'station': 'B', 'platforms': 2},
    {'id': 'BLOCK_B1', 'type': 'block', 'name': 'Block B1'},
    {'id': 'BLOCK_B2', 'type': 'block', 'name': 'Block B2'},
    {'id': 'STN_C', 'type': 'station', 'name': 'STA C', 'station': 'C', 'platforms': 2},
    {'id': 'STN_D', 'type': 'station', 'name': 'STA D', 'station': 'D', 'platforms': 2},
    {'id': 'BLOCK_D1', 'type': 'block', 'name': 'Block D1'},
    {'id': 'BLOCK_D2', 'type': 'block', 'name': 'Block D2'},
    {'id': 'STN_E', 'type': 'station', 'name': 'STA E', 'station': 'E', 'platforms': 2},
    {'id': 'BLOCK_D3', 'type': 'block', 'name': 'Block D3'},
    {'id': 'BLOCK_D4', 'type': 'block', 'name': 'Block D4'},
    {'id': 'BLOCK_D5', 'type': 'block', 'name': 'Block D5'},
    {'id': 'BLOCK_V_D2_A2', 'type': 'block', 'name': 'Block (D2-A2)'},
    {'id': 'BLOCK_F1', 'type': 'block', 'name': 'Block F1'},
    {'id': 'BLOCK_F2', 'type': 'block', 'name': 'Block F2'},
    {'id': 'STN_F', 'type': 'station', 'name': 'STA F', 'station': 'F', 'platforms': 2},
]

# Graph model of your new track layout for Dijkstra's algorithm
GRAPH = {
    'STN_A': {'BLOCK_A1': 5},
    'BLOCK_A1': {'STN_A': 5, 'BLOCK_A2': 5, 'BLOCK_F1': 4},
    'BLOCK_A2': {'BLOCK_A1': 5, 'STN_B': 5, 'BLOCK_V_D2_A2': 4},
    'STN_B': {'BLOCK_A2': 5, 'BLOCK_B1': 5},
    'BLOCK_B1': {'STN_B': 5, 'BLOCK_B2': 5, 'BLOCK_D5': 4},
    'BLOCK_B2': {'BLOCK_B1': 5, 'STN_C': 5},
    'STN_C': {'BLOCK_B2': 5},
    'STN_D': {'BLOCK_D1': 5},
    'BLOCK_D1': {'STN_D': 5, 'BLOCK_D2': 5},
    'BLOCK_D2': {'BLOCK_D1': 5, 'STN_E': 4, 'BLOCK_D3': 5, 'BLOCK_V_D2_A2': 4},
    'STN_E': {'BLOCK_D2': 4},
    'BLOCK_D3': {'BLOCK_D2': 5, 'BLOCK_D4': 5},
    'BLOCK_D4': {'BLOCK_D3': 5, 'BLOCK_D5': 4},
    'BLOCK_D5': {'BLOCK_D4': 4, 'BLOCK_B1': 4},
    'BLOCK_V_D2_A2': {'BLOCK_D2': 4, 'BLOCK_A2': 4},
    'BLOCK_F1': {'BLOCK_A1': 4, 'BLOCK_F2': 4},
    'BLOCK_F2': {'BLOCK_F1': 4, 'STN_F': 4},
    'STN_F': {'BLOCK_F2': 4},
}


class TrafficControlSystem:
    def __init__(self):
        self.trains = {}
        self.block_occupancy = {}
        self.station_platforms = {}
        self.signals = {}  # Manages signal states for junctions
        self.simulation_time = 0
        self.is_running = False
        self.train_progress = {}
        self.websocket_connections: Set[WebSocket] = set()
        
        # Initialize occupancy and signals
        for section in TRACK_SECTIONS:
            sec_id = section['id']
            if section['type'] == 'block':
                self.block_occupancy[sec_id] = None
                # Blocks that connect multiple lines are junctions for signals
                if sec_id.startswith('BLOCK_V') or sec_id == 'BLOCK_D5':
                     self.signals[sec_id] = 'green'
            elif section['type'] == 'station':
                self.station_platforms[sec_id] = {i: None for i in range(1, section['platforms'] + 1)}
                # Stations with multiple connections get a signal
                if len(GRAPH.get(sec_id, {})) > 1:
                    self.signals[sec_id] = 'green'

    async def add_websocket(self, websocket: WebSocket):
        self.websocket_connections.add(websocket)

    async def remove_websocket(self, websocket: WebSocket):
        self.websocket_connections.discard(websocket)

    async def broadcast_state(self):
        if not self.websocket_connections:
            return
        state = self.get_system_state()
        disconnected = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_text(json.dumps(state))
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.websocket_connections.discard(ws)

    def find_shortest_path(self, start_node: str, end_node: str) -> List[str]:
        """Dijkstra's algorithm to find the shortest path based on GRAPH."""
        distances = {node: float('inf') for node in GRAPH}
        distances[start_node] = 0
        pq = [(0, start_node)]
        prev_nodes = {node: None for node in GRAPH}

        while pq:
            dist, current = heapq.heappop(pq)
            if dist > distances[current]:
                continue
            if current == end_node:
                break
            for neighbor, weight in GRAPH[current].items():
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    prev_nodes[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        path = []
        current = end_node
        while current is not None:
            path.insert(0, current)
            current = prev_nodes[current]
        
        return path if path and path[0] == start_node else []

    def is_section_available(self, section_id: str, train_id: str) -> bool:
        if section_id in self.block_occupancy:
            return self.block_occupancy[section_id] is None
        elif section_id in self.station_platforms:
            return any(p is None for p in self.station_platforms[section_id].values())
        return False

    def occupy_section(self, section_id: str, train_id: str):
        if section_id in self.block_occupancy:
            self.block_occupancy[section_id] = train_id
        elif section_id in self.station_platforms:
            for p_num, occupant in self.station_platforms[section_id].items():
                if occupant is None:
                    self.station_platforms[section_id][p_num] = train_id
                    break

    def release_section(self, section_id: str, train_id: str):
        if section_id in self.block_occupancy:
            if self.block_occupancy[section_id] == train_id:
                self.block_occupancy[section_id] = None
        elif section_id in self.station_platforms:
            for p_num, occupant in self.station_platforms[section_id].items():
                if occupant == train_id:
                    self.station_platforms[section_id][p_num] = None
                    break

    def calculate_travel_time(self, train_speed: int) -> int:
        base_time = 5
        speed_factor = max(0.5, min(2.0, 100 / train_speed))
        return max(3, int(base_time * speed_factor))

    def add_train(self, train_data: dict):
        train_id = train_data['id']
        start_station = f"STN_{train_data['start']}"
        destination_station = f"STN_{train_data['destination']}"

        route = self.find_shortest_path(start_station, destination_station)
        if not route:
            logger.error(f"No route found for train {train_id} from {start_station} to {destination_station}")
            return None

        train = {
            'id': train_id, 'name': train_data['name'], 'number': train_data['number'],
            'section': route[0], 'speed': train_data['speed'], 'destination': train_data['destination'],
            'status': 'Scheduled', 'statusType': 'scheduled', 'route': route,
            'departureTime': train_data.get('departureTime', 0), 'waitingForBlock': False,
        }
        
        self.trains[train_id] = train
        self.train_progress[train_id] = {'currentRouteIndex': 0, 'lastMoveTime': 0, 'waitingForSection': None}
        self.occupy_section(route[0], train_id)
        return train

    async def update_simulation(self):
        if not self.is_running: return
        self.simulation_time += 1
        for train_id in list(self.trains.keys()):
            await self._update_train(train_id)
        self._update_signals()
        await self.broadcast_state()

    async def _update_train(self, train_id: str):
        train = self.trains[train_id]
        progress = self.train_progress[train_id]

        if self.simulation_time < train['departureTime']:
            train['status'] = f"Departs at {train['departureTime']}"
            return
        
        train['statusType'] = 'running'
        
        current_idx = progress['currentRouteIndex']
        if current_idx >= len(train['route']) - 1:
            train['status'] = 'Arrived at Destination'
            train['statusType'] = 'completed'
            return

        current_section = train['route'][current_idx]
        next_section = train['route'][current_idx + 1]
        
        time_in_section = self.simulation_time - progress['lastMoveTime']
        required_time = self.calculate_travel_time(train['speed'])

        if time_in_section >= required_time:
            if self.is_section_available(next_section, train_id):
                self.release_section(current_section, train_id)
                self.occupy_section(next_section, train_id)
                
                train['section'] = next_section
                train['status'] = f"En route to {next_section.split('_')[-1]}"
                train['waitingForBlock'] = False
                
                progress['currentRouteIndex'] += 1
                progress['lastMoveTime'] = self.simulation_time
                progress['waitingForSection'] = None
            else:
                train['waitingForBlock'] = True
                train['status'] = f"Waiting for {next_section}"
                progress['waitingForSection'] = next_section

    def _update_signals(self):
        for sig_id in self.signals: self.signals[sig_id] = 'green'
        for train in self.trains.values():
            if train['waitingForBlock'] and train['section'] in self.signals:
                self.signals[train['section']] = 'red'

    def reset_simulation(self):
        self.is_running = False
        self.simulation_time = 0
        self.trains.clear()
        self.train_progress.clear()
        
        for section_id in self.block_occupancy: self.block_occupancy[section_id] = None
        for station_id in self.station_platforms:
            for p_num in self.station_platforms[station_id]:
                self.station_platforms[station_id][p_num] = None
        
        add_default_trains(self)
        logger.info("Simulation reset")

    def start_simulation(self): self.is_running = True
    def pause_simulation(self): self.is_running = False

    def get_system_state(self):
        return {
            "trains": list(self.trains.values()),
            "blockOccupancy": self.block_occupancy,
            "stationPlatforms": self.station_platforms,
            "simulationTime": self.simulation_time,
            "isRunning": self.is_running,
            "trainProgress": self.train_progress,
            "signals": self.signals
        }

# --- Global Instance and API ---
traffic_system = TrafficControlSystem()

class TrainData(BaseModel):
    id: str; name: str; number: str; start: str
    destination: str; speed: int; departureTime: int = 0
class SimulationControl(BaseModel):
    action: str

# ===================================================================
# === THIS IS THE ONLY SECTION THAT HAS BEEN CHANGED ===
def add_default_trains(system: TrafficControlSystem):
    """Adds a more complex and conflicting set of trains to test the system."""
    default_trains = [
        # Fast express on a common path
        {'id': 'T1', 'name': 'High-Speed Express', 'number': 'X101', 'start': 'D', 'destination': 'C', 'speed': 140, 'departureTime': 0},
        
        # Slower goods train on a long, disruptive path that crosses the main line
        {'id': 'T2', 'name': 'Heavy Cargo', 'number': 'G550', 'start': 'F', 'destination': 'E', 'speed': 50, 'departureTime': 2},
        
        # A local train that will likely have to wait for the express and cargo trains
        {'id': 'T3', 'name': 'Regional Commuter', 'number': 'R303', 'start': 'A', 'destination': 'C', 'speed': 80, 'departureTime': 4},
        
        # An opposing express train that will conflict with the cargo train
        {'id': 'T4', 'name': 'Inter-City Sprinter', 'number': 'X202', 'start': 'E', 'destination': 'A', 'speed': 130, 'departureTime': 8},
        
        # Another local train on a short but potentially congested route
        {'id': 'T5', 'name': 'B-Line Shuttle', 'number': 'S415', 'start': 'B', 'destination': 'D', 'speed': 70, 'departureTime': 12},
    ]
    for train_data in default_trains:
        system.add_train(train_data)
# ===================================================================

@app.on_event("startup")
async def startup_event():
    add_default_trains(traffic_system)
    asyncio.create_task(simulation_loop())

async def simulation_loop():
    while True:
        if traffic_system.is_running:
            await traffic_system.update_simulation()
        await asyncio.sleep(1.5)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await traffic_system.add_websocket(websocket)
    try:
        await websocket.send_text(json.dumps(traffic_system.get_system_state()))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await traffic_system.remove_websocket(websocket)

@app.post("/simulation-control")
async def control_simulation(control: SimulationControl):
    if control.action == "start": traffic_system.start_simulation()
    elif control.action == "pause": traffic_system.pause_simulation()
    elif control.action == "reset": traffic_system.reset_simulation()
    else: raise HTTPException(status_code=400, detail="Invalid action")
    await traffic_system.broadcast_state()
    return {"status": "success", "action": control.action}

@app.post("/add-train")
async def add_train_endpoint(train: TrainData):
    new_train = traffic_system.add_train(train.dict())
    if new_train is None:
        raise HTTPException(status_code=400, detail="Could not create train. Route may not exist.")
    await traffic_system.broadcast_state()
    return {"status": "success", "train": new_train}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)