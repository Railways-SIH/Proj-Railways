<<<<<<< HEAD
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests, math

app = FastAPI(title="Railway Network Schematic Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
=======
import asyncio
import json
import logging
from typing import Dict, List, Set, Tuple, Any # Added Any
import heapq

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the optimization and prediction modules
from optimizer.cp_solver import solve_junction_precedence
from predictor.mock_predictor import run_predictive_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Setup ---
app = FastAPI(title="Intelligent Railway Control Backend", version="4.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
>>>>>>> branch/santhosh
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
# -------------------------
# Station list (input)
# -------------------------
STATIONS_INPUT = [
    {"id": "SC", "name": "Secunderabad Jn", "lat": 17.4365, "lon": 78.4983, "platforms": 6},
    {"id": "MJF", "name": "Malkajgiri",      "lat": 17.4521, "lon": 78.5230, "platforms": 2},
    {"id": "BMO", "name": "Bolarum",         "lat": 17.5247, "lon": 78.5238, "platforms": 2},
    {"id": "MDF", "name": "Medchal",         "lat": 17.6013, "lon": 78.6185, "platforms": 3},
]

# -------------------------
# Helper to fetch OSM (can extend later)
# -------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def fetch_osm_network(stations):
    # for now we just return synthetic — you can extend to OSM parsing later
    return {
        "stations": stations,
        "tracks": [], "signals": [], "level_crossings": [], "junctions": []
    }

# -------------------------
# Build schematic layout
# -------------------------
@app.get("/schematic")
def get_schematic(osm: bool = False):
    network = fetch_osm_network(STATIONS_INPUT) if osm else {"stations": STATIONS_INPUT}

    spacing_x = 200   # distance between stations
    offset_x = 100
    offset_y = 200

    schematic_stations = []
    schematic_blocks = []
    schematic_signals = []
    schematic_crossings = []
    schematic_junctions = []

    # Place stations linearly
    for i, st in enumerate(network["stations"]):
        x = offset_x + i * spacing_x
        y = offset_y
        schematic_stations.append({
            "id": st["id"], "name": st["name"], "platforms": st.get("platforms", 1),
            "x": x, "y": y
        })

        # create block between previous and current station
        if i > 0:
            prev = schematic_stations[i-1]
            block_id = f"BLOCK_{prev['id']}_{st['id']}"
            block_x = prev["x"] + 40
            block_y = y - 10
            block_w = spacing_x - 80
            block_h = 20

            schematic_blocks.append({
                "id": block_id, "from": prev["id"], "to": st["id"],
                "status": "free", "x": block_x, "y": block_y,
                "width": block_w, "height": block_h
            })

            # add one signal per block
            schematic_signals.append({
                "id": f"SIGNAL_{block_id}",
                "x": block_x + block_w // 2,
                "y": block_y - 30,
                "aspect": "RED"
            })

            # demo crossing under the middle block
            if i == 2:  # just an example
                schematic_crossings.append({
                    "id": f"CROSS_{block_id}",
                    "x": block_x + block_w // 2,
                    "y": block_y + 60,
                    "status": "open"
                })

    return {
        "stations": schematic_stations,
        "blocks": schematic_blocks,
        "signals": schematic_signals,
        "crossings": schematic_crossings,
        "junctions": schematic_junctions,
        "trains": [],
        "canvas": {
            "width": offset_x + len(STATIONS_INPUT) * spacing_x,
            "height": offset_y + 200
        }
    }

@app.get("/")
def root():
    return {"message": "Railway schematic backend running"}


@app.get("/stations")
def get_stations():
    return {"stations": STATIONS_INPUT}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
=======
# --- Static Network Data (Pulled from Frontend for Consistency) ---
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
# Helper to map Train ID to its static properties
TRAIN_STATIC_DATA: Dict[str, Dict[str, Any]] = {}

class TrafficControlSystem:
    def __init__(self):
        self.trains: Dict[str, Dict[str, Any]] = {}
        self.block_occupancy: Dict[str, Any] = {}
        self.station_platforms: Dict[str, Dict[int, Any]] = {}
        self.simulation_time = 0
        self.is_running = False
        self.train_progress: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Set[WebSocket] = set()
        self.metrics = { "throughput": 0, "avgDelay": 0, "utilization": 0, "avgSpeed": 0 }
        self.completed_train_stats = []
        self.events: List[str] = []
        
        # --- NEW AI/OR STATE ---
        self.optimal_schedule: Dict[str, int] = {} # {train_id: scheduled_entry_time_tick}
        self.last_or_solve_time = 0
        self.ai_metrics: Dict[str, Any] = {} # Stores prediction data
        self.last_ai_predict_time = 0

        for section in TRACK_SECTIONS:
            sec_id = section['id']
            if section['type'] == 'block': self.block_occupancy[sec_id] = None
            elif section['type'] == 'station': self.station_platforms[sec_id] = {i: None for i in range(1, section['platforms'] + 1)}

    async def add_websocket(self, websocket: WebSocket): self.websocket_connections.add(websocket)
    async def remove_websocket(self, websocket: WebSocket): self.websocket_connections.discard(websocket)

    async def broadcast_state(self):
        if not self.websocket_connections: return
        state = self.get_system_state()
        state_json = json.dumps(state, default=str) # Use default=str for safety with datetimes/enums
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
     
    def add_train(self, train_data: Dict[str, Any]):
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
            'priority': train_data.get('priority', 99)
        }
        self.trains[train_id] = train
        self.train_progress[train_id] = {'currentRouteIndex': 0, 'lastMoveTime': train_data.get('departureTime', 0)}
        self.occupy_section(route[0], train_id)
        
        # Store static data for OR/AI modules
        TRAIN_STATIC_DATA[train_id] = {'name': train_data['name'], 'priority': train_data['priority'], 'speed': train_data['speed']}
        
        return train
    
    def _run_ai_prediction(self):
        """Runs the mock AI predictor to get estimated delays and proactive advice."""
        # Only re-run the predictor every 15 ticks, or if the initial run hasn't happened.
        if self.simulation_time - self.last_ai_predict_time >= 15 or not self.ai_metrics:
            
            # --- CRITICAL FIX: Run prediction FIRST ---
            new_ai_metrics = run_predictive_analysis(list(self.trains.values()), self.optimal_schedule)
            self.ai_metrics = new_ai_metrics # Update the instance variable
            self.last_ai_predict_time = self.simulation_time
            
            # --- Safety check before accessing keys ---
            if (self.ai_metrics and 
                'ai_recommendation' in self.ai_metrics and 
                self.ai_metrics['ai_recommendation']['action_type'] != 'NONE'):
                self.events.append(f"AI: Predicted {self.ai_metrics['delay_saving_potential']} min delay saving by {self.ai_metrics['ai_recommendation']['action_type']}")

    def _run_reoptimization(self):
        """Runs the OR solver for conflict resolution in the critical junction."""
        # Only run the OR solver every 30 seconds of simulation time, or if a critical conflict is known.
        if self.simulation_time - self.last_or_solve_time >= 30:
            current_state_for_or = self.get_system_state()
            # Only pass relevant trains/blocks to the specialized junction solver
            result = solve_junction_precedence(current_state_for_or, TRAIN_STATIC_DATA, self.simulation_time)
            
            if result['recommendation'] == 'OPTIMAL_SCHEDULE_FOUND':
                # Update the optimal schedule with the new precedence plan
                new_schedule = {
                    train_id: int(decision['start_sim_time_seconds']) 
                    for train_id, decision in result['decisions'].items()
                }
                self.optimal_schedule = new_schedule
                self.events.append("OR: New conflict-free schedule generated.")
            else:
                self.optimal_schedule = {} # Clear schedule if no conflict found
            
            self.last_or_solve_time = self.simulation_time

    def _update_metrics(self):
        completed_count = len(self.completed_train_stats)
        self.metrics["throughput"] = (completed_count / self.simulation_time) * 3600 if self.simulation_time > 0 else 0 # Converted to hours
        
        total_actual_time = sum(s['actual_time'] for s in self.completed_train_stats)
        total_ideal_time = sum(s['ideal_time'] for s in self.completed_train_stats)
        total_delay = total_actual_time - total_ideal_time
        
        # NOTE: Avg Delay now reflects difference between actual and ideal time
        self.metrics["avgDelay"] = total_delay / completed_count if completed_count > 0 else 0
        
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
            
            # 1. Run Intelligence Modules
            self._run_ai_prediction() 
            self._run_reoptimization()
            
            # 2. Update all trains
            await asyncio.gather(*(self._update_train(tid) for tid in list(self.trains.keys())))
            
            # 3. Update Metrics & Broadcast
            self._update_metrics()
            await self.broadcast_state()

    async def _update_train(self, train_id: str):
        train = self.trains[train_id]
        progress = self.train_progress[train_id]
        was_waiting = train['waitingForBlock']
        
        # 1. Skip if scheduled to start later or already completed
        if self.simulation_time < train['departureTime']: return
        if train['statusType'] == 'completed': return
        train['statusType'] = 'running'
        
        current_idx = progress['currentRouteIndex']
        
        # 2. Handle Arrival
        if current_idx >= len(train['route']) - 1:
            train.update({'status': 'Arrived', 'statusType': 'completed'})
            self.completed_train_stats.append({'id': train_id, 'ideal_time': train['idealTravelTime'], 'actual_time': self.simulation_time - train['departureTime']})
            self.release_section(train['section'], train_id)
            self.events.append(f"Arrival: {train['number']}. Total delay: {self.simulation_time - train['departureTime'] - train['idealTravelTime']}s")
            return
        
        current_section, next_section = train['route'][current_idx], train['route'][current_idx + 1]
        is_at_station, is_stop = current_section.startswith('STN_'), current_section in train.get('stops', [])

        # 3. Handle Dwell Time at Stop Stations
        if is_at_station and is_stop:
            if not train['atStation']: 
                train['atStation'], train['dwellTimeStart'] = True, self.simulation_time
                self.events.append(f"Halt: {train['number']} at {current_section}.")
            if self.simulation_time - train['dwellTimeStart'] < 5: # 5 second dwell time
                train['status'] = f"Halting at {current_section}"; return
        if not current_section.startswith('STN_') and train['atStation']: train['atStation'] = False
        
        required_time = self.calculate_travel_time(train['speed'], is_at_station and not is_stop)
        
        # 4. Check for Movement Eligibility
        if self.simulation_time - progress['lastMoveTime'] >= required_time:
            
            can_move_by_or = True
            
            # A. OR Precedence Check: Is this train *scheduled* to move now?
            if train_id in self.optimal_schedule:
                optimal_entry_time = self.optimal_schedule.get(train_id)
                if self.simulation_time < optimal_entry_time:
                    can_move_by_or = False
                elif self.simulation_time >= optimal_entry_time:
                    # Time to move, clear the schedule entry for the next re-solve
                    del self.optimal_schedule[train_id] 
            
            # B. Safety Check: Is the next block clear?
            is_next_section_available = self.is_section_available(next_section, train_id)

            if not is_next_section_available:
                # Blocked by another train (Safety constraint)
                train.update({'waitingForBlock': True, 'status': f"Blocked by {self.block_occupancy.get(next_section, 'other train')}"})
                return

            if not can_move_by_or:
                # OR is forcing a hold (Precedence constraint)
                train.update({'waitingForBlock': True, 'status': f"Holding for OR Plan at {current_section}"})
                if not was_waiting:
                    self.events.append(f"OR Hold: {train['number']} holds for higher priority train.")
                return

            # 5. Move Train (Both OR/Precedence and Safety checks passed)
            self.release_section(current_section, train_id)
            self.occupy_section(next_section, train_id)
            train.update({'section': next_section, 'status': f"En route to {next_section}", 'waitingForBlock': False})
            progress.update({'currentRouteIndex': current_idx + 1, 'lastMoveTime': self.simulation_time})

    def get_system_state(self) -> Dict[str, Any]: 
        """Gathers all necessary state data for the frontend broadcast."""
        return { 
            "trains": list(self.trains.values()), 
            "blockOccupancy": self.block_occupancy, 
            "stationPlatforms": self.station_platforms, 
            "simulationTime": self.simulation_time, 
            "isRunning": self.is_running, 
            "trainProgress": self.train_progress, 
            "metrics": self.metrics, 
            "events": self.events,
            "aiMetrics": self.ai_metrics, # ALWAYS send AI metrics
            "orDecisions": self.optimal_schedule # ALWAYS send OR decisions
        }

    def find_shortest_path(self, start_node: str, end_node: str) -> List[str]:
        distances: Dict[str, float] = {node: float('inf') for node in GRAPH}; distances[start_node]=0
        pq: List[Tuple[float, str]] = [(0, start_node)]; prev_nodes: Dict[str, Any] = {node: None for node in GRAPH}
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
        self.optimal_schedule.clear() # Clear OR state
        self.ai_metrics.clear() # Clear AI state
        for sec_id in self.block_occupancy: self.block_occupancy[sec_id] = None
        for stn_id in self.station_platforms:
            for p_num in self.station_platforms[stn_id]: self.station_platforms[stn_id][p_num] = None
        add_default_trains(self); logger.info("Simulation reset")
         
    def start_simulation(self): self.is_running = True
    def pause_simulation(self): self.is_running = False

traffic_system = TrafficControlSystem()
class TrainData(BaseModel): id: str; name: str; number: str; start: str; destination: str; speed: int; departureTime: int = 0; stops: List[str] = []; priority: int = 99
class SimulationControl(BaseModel): action: str
class OverrideControl(BaseModel): action: str; train_id: str; value: Any = None # New model for override

def add_default_trains(system: TrafficControlSystem):
    # Higher priority trains have lower numbers (e.g., Express=10)
    default_trains = [
        {'id': 'T1', 'number': '12301', 'name': 'Express', 'start': 'D', 'destination': 'C', 'speed': 140, 'departureTime': 0, 'stops': ['STN_D', 'STN_C'], 'priority': 10},
        {'id': 'T2', 'number': '12302', 'name': 'Cargo', 'start': 'F', 'destination': 'E', 'speed': 50, 'departureTime': 2, 'stops': ['STN_F', 'STN_E'], 'priority': 30},
        {'id': 'T3', 'number': '12303', 'name': 'Commuter', 'start': 'A', 'destination': 'C', 'speed': 80, 'departureTime': 4, 'stops': ['STN_A', 'STN_B', 'STN_C'], 'priority': 20},
        {'id': 'T4', 'number': '12304', 'name': 'Sprinter', 'start': 'E', 'destination': 'A', 'speed': 130, 'departureTime': 8, 'stops': ['STN_E', 'STN_A'], 'priority': 10},
    ]
    for train_data in default_trains: system.add_train(train_data)

@app.on_event("startup")
async def startup_event(): 
    add_default_trains(traffic_system)
    traffic_system._run_ai_prediction() # Run initial prediction to populate AI panel
    asyncio.create_task(simulation_loop())

async def simulation_loop():
    while True:
        await traffic_system.update_simulation()
        await asyncio.sleep(1.0) # Speed up simulation slightly

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept(); await traffic_system.add_websocket(websocket)
    try:
        await websocket.send_text(json.dumps(traffic_system.get_system_state(), default=str))
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

@app.post("/override-control")
async def override_control_endpoint(control: OverrideControl):
    train_id = control.train_id
    action = control.action
    value = control.value
    
    # Example logic for manual overrides (must be refined)
    if action == 'OVERRIDE_MOVE':
        # Force a train to move immediately, ignoring the OR schedule
        if train_id in traffic_system.trains:
            train = traffic_system.trains[train_id]
            # Simple override: set lastMoveTime back to force immediate eligibility
            traffic_system.train_progress[train_id]['lastMoveTime'] = traffic_system.simulation_time - 100 
            if train_id in traffic_system.optimal_schedule:
                 del traffic_system.optimal_schedule[train_id] # Clear OR constraint
            traffic_system.events.append(f"OVERRIDE: Controller manually forced {train['number']} to proceed.")

    elif action == 'ACCEPT_AI':
        # Accept AI's proactive speed adjustment suggestion (e.g., slow down for 60s)
        if train_id in traffic_system.trains:
            # In a full system, this would apply a temporary speed limit/hold
            traffic_system.events.append(f"ACCEPT: Controller accepted AI advice for {traffic_system.trains[train_id]['number']}.")
    
    await traffic_system.broadcast_state()
    return {"status": "success", "action": action}

@app.post("/add-train")
async def add_train_endpoint(train: TrainData):
    new_train = traffic_system.add_train(train.dict())
    if new_train is None: raise HTTPException(status_code=400, detail="Could not create train")
    await traffic_system.broadcast_state(); return {"status": "success", "train": new_train}

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8000)
>>>>>>> branch/santhosh
