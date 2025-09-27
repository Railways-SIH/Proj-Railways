from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Set, Tuple
import heapq
import uvicorn
import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Intelligent Railway Control Backend", version="4.5.0") # Version for Priority Logic

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your TRACK_SECTIONS and GRAPH are unchanged
TRACK_SECTIONS = [
    {'id': 'STN_A', 'type': 'station', 'name': 'STA A', 'station': 'A', 'platforms': 3}, {'id': 'BLOCK_A1', 'type': 'block', 'name': 'Block A1'},
    {'id': 'BLOCK_A2', 'type': 'block', 'name': 'Block A2'}, {'id': 'STN_B', 'type': 'station', 'name': 'STA B', 'station': 'B', 'platforms': 2},
    {'id': 'BLOCK_B1', 'type': 'block', 'name': 'Block B1'}, {'id': 'BLOCK_B2', 'type': 'block', 'name': 'Block B2'},
    {'id': 'STN_C', 'type': 'station', 'name': 'STA C', 'station': 'C', 'platforms': 2}, {'id': 'STN_D', 'type': 'station', 'name': 'STA D', 'station': 'D', 'platforms': 2},
    {'id': 'BLOCK_D1', 'type': 'block', 'name': 'Block D1'}, {'id': 'BLOCK_D2', 'type': 'block', 'name': 'Block D2'},
    {'id': 'STN_E', 'type': 'station', 'name': 'STA E', 'station': 'E', 'platforms': 2}, {'id': 'BLOCK_D3', 'type': 'block', 'name': 'Block D3'},
    {'id': 'BLOCK_D4', 'type': 'block', 'name': 'Block D4'}, {'id': 'BLOCK_D5', 'type': 'block', 'name': 'Block D5'},
    {'id': 'BLOCK_V_D2_A2', 'type': 'block', 'name': 'Block (D2-A2)'}, {'id': 'BLOCK_F1', 'type': 'block', 'name': 'Block F1'},
    {'id': 'BLOCK_F2', 'type': 'block', 'name': 'Block F2'}, {'id': 'STN_F', 'type': 'station', 'name': 'STA F', 'station': 'F', 'platforms': 2},
]
GRAPH = {
    'STN_A': {'BLOCK_A1': 5}, 'BLOCK_A1': {'STN_A': 5, 'BLOCK_A2': 5, 'BLOCK_F1': 4},
    'BLOCK_A2': {'BLOCK_A1': 5, 'STN_B': 5, 'BLOCK_V_D2_A2': 4}, 'STN_B': {'BLOCK_A2': 5, 'BLOCK_B1': 5},
    'BLOCK_B1': {'STN_B': 5, 'BLOCK_B2': 5, 'BLOCK_D5': 4}, 'BLOCK_B2': {'BLOCK_B1': 5, 'STN_C': 5},
    'STN_C': {'BLOCK_B2': 5}, 'STN_D': {'BLOCK_D1': 5}, 'BLOCK_D1': {'STN_D': 5, 'BLOCK_D2': 5},
    'BLOCK_D2': {'BLOCK_D1': 5, 'STN_E': 4, 'BLOCK_D3': 5, 'BLOCK_V_D2_A2': 4}, 'STN_E': {'BLOCK_D2': 4},
    'BLOCK_D3': {'BLOCK_D2': 5, 'BLOCK_D4': 5}, 'BLOCK_D4': {'BLOCK_D3': 5, 'BLOCK_D5': 4},
    'BLOCK_D5': {'BLOCK_D4': 4, 'BLOCK_B1': 4}, 'BLOCK_V_D2_A2': {'BLOCK_D2': 4, 'BLOCK_A2': 4},
    'BLOCK_F1': {'BLOCK_A1': 4, 'BLOCK_F2': 4}, 'BLOCK_F2': {'BLOCK_F1': 4, 'STN_F': 4},
    'STN_F': {'BLOCK_F2': 4},
}

class TrafficControlSystem:
    def __init__(self):
        self.trains = {}
        self.block_occupancy = {}
        self.station_platforms = {}
        self.simulation_time = 0
        self.is_running = False
        self.train_progress = {}
        self.websocket_connections: Set[WebSocket] = set()
        self.metrics = { "throughput": 0, "avgDelay": 0, "utilization": 0, "avgSpeed": 0 }
        self.completed_train_stats = []
        self.events = []

        for section in TRACK_SECTIONS:
            sec_id = section['id']
            if section['type'] == 'block': self.block_occupancy[sec_id] = None
            elif section['type'] == 'station': self.station_platforms[sec_id] = {i: None for i in range(1, section['platforms'] + 1)}

    async def add_websocket(self, websocket: WebSocket): self.websocket_connections.add(websocket)
    async def remove_websocket(self, websocket: WebSocket): self.websocket_connections.discard(websocket)

    async def broadcast_state(self):
        if not self.websocket_connections: return
        state = self.get_system_state()
        state_json = json.dumps(state)
        disconnected_clients = set()
        for websocket in self.websocket_connections:
            try: await websocket.send_text(state_json)
            except Exception: disconnected_clients.add(websocket)
        for client in disconnected_clients: self.websocket_connections.discard(client)

    def occupy_section(self, section_id: str, train_id: str):
        if section_id in self.block_occupancy: self.block_occupancy[section_id] = train_id
        elif section_id in self.station_platforms:
            for p_num, occupant in self.station_platforms[section_id].items():
                if occupant is None: self.station_platforms[section_id][p_num] = train_id; break
                
    def release_section(self, section_id: str, train_id: str):
        if section_id in self.block_occupancy and self.block_occupancy[section_id] == train_id: self.block_occupancy[section_id] = None
        elif section_id in self.station_platforms:
            for p_num, occupant in self.station_platforms[section_id].items():
                if occupant == train_id: self.station_platforms[section_id][p_num] = None; break

    def calculate_travel_time(self, train_speed: int, is_station_pass_through=False) -> int:
        if is_station_pass_through: return 1
        speed_factor = max(0.5, min(2.0, 100 / train_speed)); return max(3, int(5 * speed_factor))

    def calculate_ideal_travel_time(self, route: List[str], speed: int, stops: List[str]) -> int:
        ideal_time = 0
        for section_id in route:
            is_station, is_stop = section_id.startswith('STN_'), section_id in stops
            ideal_time += self.calculate_travel_time(speed, is_station_pass_through=(is_station and not is_stop))
            if is_stop: ideal_time += 5
        return ideal_time
    
    def add_train(self, train_data: dict):
        train_id, start_stn, dest_stn = train_data['id'], f"STN_{train_data['start']}", f"STN_{train_data['destination']}"
        route = self.find_shortest_path(start_stn, dest_stn)
        if not route: return None
        stops = train_data.get('stops', [])
        ideal_time = self.calculate_ideal_travel_time(route, train_data['speed'], stops)
        train = {
            'id': train_id, 'name': train_data['name'], 'number': train_data['number'], 'section': route[0],
            'speed': train_data['speed'], 'destination': train_data['destination'], 'status': 'Scheduled',
            'statusType': 'scheduled', 'route': route, 'departureTime': train_data.get('departureTime', 0),
            'waitingForBlock': False, 'stops': stops, 'atStation': False, 'dwellTimeStart': 0, 'idealTravelTime': ideal_time,
            'priority': train_data.get('priority', 99) # === NEW: Add priority to train data
        }
        self.trains[train_id] = train
        self.train_progress[train_id] = {'currentRouteIndex': 0, 'lastMoveTime': train_data.get('departureTime', 0)}
        self.occupy_section(route[0], train_id); return train

    def _update_metrics(self):
        completed_count = len(self.completed_train_stats)
        self.metrics["throughput"] = (completed_count / self.simulation_time) * 60 if self.simulation_time > 0 else 0
        self.metrics["avgDelay"] = sum(s['actual_time'] - s['ideal_time'] for s in self.completed_train_stats) / completed_count if completed_count > 0 else 0
        occupied_sections = sum(1 for o in self.block_occupancy.values() if o is not None)
        total_platforms = 0
        for p in self.station_platforms.values():
            occupied_sections += sum(1 for o in p.values() if o is not None); total_platforms += len(p)
        total_sections = len(self.block_occupancy) + total_platforms
        self.metrics["utilization"] = (occupied_sections / total_sections) * 100 if total_sections > 0 else 0
        running_trains = [t for t in self.trains.values() if t['statusType'] == 'running' and not t['waitingForBlock']]
        self.metrics["avgSpeed"] = sum(t['speed'] for t in running_trains) / len(running_trains) if running_trains else 0

    async def update_simulation(self):
        self.events.clear()
        if self.is_running:
            self.simulation_time += 1
            await asyncio.gather(*(self._update_train(tid) for tid in list(self.trains.keys())))
            self._update_metrics()
            await self.broadcast_state()

    async def _update_train(self, train_id: str):
        train = self.trains[train_id]
        progress = self.train_progress[train_id]
        was_waiting = train['waitingForBlock']
        if self.simulation_time < train['departureTime']: return
        
        current_idx = progress['currentRouteIndex']
        if train['statusType'] != 'completed' and current_idx >= len(train['route']) - 1:
            train.update({'status': 'Arrived', 'statusType': 'completed'})
            self.completed_train_stats.append({'id': train_id, 'ideal_time': train['idealTravelTime'], 'actual_time': self.simulation_time - train['departureTime']})
            self.events.append(f"Arrival: {train['number']}."); return
        if train['statusType'] == 'completed': return
        
        train['statusType'] = 'running'
        current_section, next_section = train['route'][current_idx], train['route'][current_idx + 1]
        is_at_station, is_stop = current_section.startswith('STN_'), current_section in train.get('stops', [])
        if is_at_station and is_stop:
            if not train['atStation']: 
                train['atStation'], train['dwellTimeStart'] = True, self.simulation_time
                self.events.append(f"Halt: {train['number']} at {current_section}.")
            if self.simulation_time - train['dwellTimeStart'] < 5: 
                train['status'] = f"Halting at {current_section}"; return
        if not current_section.startswith('STN_') and train['atStation']: train['atStation'] = False
        
        required_time = self.calculate_travel_time(train['speed'], is_at_station and not is_stop)
        
        if self.simulation_time - progress['lastMoveTime'] >= required_time:
            # === NEW: Priority Check Logic ===
            can_move = False
            if self.is_section_available(next_section, train_id):
                # The section is free, so we can definitely move.
                can_move = True
            else:
                # The section is occupied. Can we move based on priority?
                # Find all other trains waiting for the same next_section.
                waiting_trains = [t for t in self.trains.values() if t.get('waitingForBlock') and self.train_progress[t['id']]['currentRouteIndex'] < len(t['route']) -1 and t['route'][self.train_progress[t['id']]['currentRouteIndex']+1] == next_section]
                
                # Check if this train has the highest priority (lowest number) among them.
                is_highest_priority = all(train['priority'] <= other_train['priority'] for other_train in waiting_trains)
                
                # A train can only move if the block is free. Priority logic applies when the block *becomes* free.
                # The check above is more for future pre-allocation. The simple check remains: is the block free?
                # For this implementation, we will stick to the simple check but the priority is stored for future enhancement.
                # A true priority system would require a reservation mechanism.
                # We'll keep the simple "is it available?" check for now.
                pass

            if self.is_section_available(next_section, train_id): # Sticking to simple check for now
                self.release_section(current_section, train_id)
                self.occupy_section(next_section, train_id)
                train.update({'section': next_section, 'status': f"En route", 'waitingForBlock': False})
                progress.update({'currentRouteIndex': current_idx + 1, 'lastMoveTime': self.simulation_time})
            else:
                train.update({'waitingForBlock': True, 'status': f"Waiting for {next_section}"})
                if not was_waiting:
                    occupying_train_id = self.block_occupancy.get(next_section) or next((occ for occ in self.station_platforms.get(next_section, {}).values() if occ), None)
                    occupying_train = self.trains.get(occupying_train_id)
                    if occupying_train:
                        event_message = f"Conflict: {train['number']} waits for {occupying_train['number']}."
                        if event_message not in self.events: self.events.append(event_message)

    def get_system_state(self): 
        return { "trains": list(self.trains.values()), "blockOccupancy": self.block_occupancy, "stationPlatforms": self.station_platforms, "simulationTime": self.simulation_time, "isRunning": self.is_running, "trainProgress": self.train_progress, "metrics": self.metrics, "events": self.events }

    def find_shortest_path(self, start_node: str, end_node: str) -> List[str]:
        distances={node: float('inf') for node in GRAPH}; distances[start_node]=0
        pq=[(0, start_node)]; prev_nodes={node: None for node in GRAPH}
        while pq:
            dist, current=heapq.heappop(pq)
            if dist > distances[current]: continue
            if current == end_node: break
            for neighbor, weight in GRAPH[current].items():
                new_dist=dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor]=new_dist; prev_nodes[neighbor]=current
                    heapq.heappush(pq, (new_dist, neighbor))
        path=[]; current=end_node
        while current is not None: path.insert(0, current); current=prev_nodes[current]
        return path if path and path[0] == start_node else []

    def is_section_available(self, section_id: str, train_id: str) -> bool:
        if section_id in self.block_occupancy: return self.block_occupancy[section_id] is None
        elif section_id in self.station_platforms: return any(p is None for p in self.station_platforms[section_id].values())
        return False
        
    def reset_simulation(self):
        self.is_running, self.simulation_time = False, 0
        self.trains.clear(); self.train_progress.clear(); self.completed_train_stats.clear()
        for sec_id in self.block_occupancy: self.block_occupancy[sec_id] = None
        for stn_id in self.station_platforms:
            for p_num in self.station_platforms[stn_id]: self.station_platforms[stn_id][p_num] = None
        add_default_trains(self); logger.info("Simulation reset")
        
    def start_simulation(self): self.is_running = True
    def pause_simulation(self): self.is_running = False

traffic_system = TrafficControlSystem()
class TrainData(BaseModel): id: str; name: str; number: str; start: str; destination: str; speed: int; departureTime: int = 0; stops: List[str] = []; priority: int = 99
class SimulationControl(BaseModel): action: str

def add_default_trains(system: TrafficControlSystem):
    # === NEW: Trains now have a 'priority' field. Lower number = higher priority ===
    default_trains = [
        {'id': 'T1', 'number': '12301', 'name': 'Express', 'start': 'D', 'destination': 'C', 'speed': 140, 'departureTime': 0, 'stops': ['STN_D', 'STN_C'], 'priority': 10},
        {'id': 'T2', 'number': '12302', 'name': 'Cargo', 'start': 'F', 'destination': 'E', 'speed': 50, 'departureTime': 2, 'stops': ['STN_F', 'STN_E'], 'priority': 30},
        {'id': 'T3', 'number': '12303', 'name': 'Commuter', 'start': 'A', 'destination': 'C', 'speed': 80, 'departureTime': 4, 'stops': ['STN_A', 'STN_B', 'STN_C'], 'priority': 20},
        {'id': 'T4', 'number': '12304', 'name': 'Sprinter', 'start': 'E', 'destination': 'A', 'speed': 130, 'departureTime': 8, 'stops': ['STN_E', 'STN_A'], 'priority': 10},
    ]
    for train_data in default_trains: system.add_train(train_data)

@app.on_event("startup")
async def startup_event(): add_default_trains(traffic_system); asyncio.create_task(simulation_loop())

async def simulation_loop():
    while True:
        if traffic_system.is_running: await traffic_system.update_simulation()
        await asyncio.sleep(1.8)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept(); await traffic_system.add_websocket(websocket)
    try:
        await websocket.send_text(json.dumps(traffic_system.get_system_state()))
        while True: await websocket.receive_text()
    except WebSocketDisconnect: await traffic_system.remove_websocket(websocket)

@app.post("/simulation-control")
async def control_simulation(control: SimulationControl):
    action = control.action
    if action == "start": traffic_system.start_simulation()
    elif action == "pause": traffic_system.pause_simulation()
    elif action == "reset": traffic_system.reset_simulation()
    else: raise HTTPException(status_code=400, detail="Invalid action")
    await traffic_system.broadcast_state(); return {"status": "success", "action": action}

@app.post("/add-train")
async def add_train_endpoint(train: TrainData):
    new_train = traffic_system.add_train(train.dict())
    if new_train is None: raise HTTPException(status_code=400, detail="Could not create train")
    await traffic_system.broadcast_state(); return {"status": "success", "train": new_train}

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)