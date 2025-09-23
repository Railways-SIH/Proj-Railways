from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional, Set
import heapq
import uvicorn
import asyncio
import json
from datetime import datetime, timedelta
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized Railway Traffic Control Backend", version="3.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Track sections mapping to match frontend layout
TRACK_SECTIONS = [
    # Main horizontal line (A -> B -> C)
    {'id': 'STN_A', 'type': 'station', 'name': 'STA A', 'station': 'A', 'platforms': 3},
    {'id': 'BLOCK_A1', 'type': 'block', 'name': 'Block A1', 'station': None},
    {'id': 'BLOCK_A2', 'type': 'block', 'name': 'Block A2', 'station': None},
    {'id': 'STN_B', 'type': 'station', 'name': 'STA B', 'station': 'B', 'platforms': 2},
    {'id': 'BLOCK_B1', 'type': 'block', 'name': 'Block B1', 'station': None},
    {'id': 'BLOCK_B2', 'type': 'block', 'name': 'Block B2', 'station': None},
    {'id': 'STN_C', 'type': 'station', 'name': 'STA C', 'station': 'C', 'platforms': 2},

    # Upper branch (D -> E -> junction with main line)
    {'id': 'STN_D', 'type': 'station', 'name': 'STA D', 'station': 'D', 'platforms': 2},
    {'id': 'BLOCK_D1', 'type': 'block', 'name': 'Block D1', 'station': None},
    {'id': 'BLOCK_D2', 'type': 'block', 'name': 'Block D2', 'station': None},
    {'id': 'STN_E', 'type': 'station', 'name': 'STA E', 'station': 'E', 'platforms': 2},
    {'id': 'BLOCK_D3', 'type': 'block', 'name': 'Block D3', 'station': None},
    {'id': 'BLOCK_D4', 'type': 'block', 'name': 'Block D4', 'station': None},
    
    # Junction blocks
    {'id': 'BLOCK_D5', 'type': 'block', 'name': 'Block D5', 'station': None},
    {'id': 'BLOCK_V_D2_A2', 'type': 'block', 'name': 'Block (D2-A2)', 'station': None},
    
    # Lower branch (A1 -> F)
    {'id': 'BLOCK_F1', 'type': 'block', 'name': 'Block F1', 'station': None},
    {'id': 'BLOCK_F2', 'type': 'block', 'name': 'Block F2', 'station': None},
    {'id': 'STN_F', 'type': 'station', 'name': 'STA F', 'station': 'F', 'platforms': 2},
]

# Enhanced route definitions with multiple path options
RAILWAY_NETWORK = {
    # Direct connections between sections
    'STN_A': ['BLOCK_A1'],
    'BLOCK_A1': ['STN_A', 'BLOCK_A2', 'BLOCK_F1'],  # Junction to lower branch
    'BLOCK_A2': ['BLOCK_A1', 'STN_B', 'BLOCK_V_D2_A2'],  # Junction to upper branch
    'STN_B': ['BLOCK_A2', 'BLOCK_B1'],
    'BLOCK_B1': ['STN_B', 'BLOCK_B2', 'BLOCK_D5'],  # Junction from upper branch
    'BLOCK_B2': ['BLOCK_B1', 'STN_C'],
    'STN_C': ['BLOCK_B2'],
    
    # Upper branch network
    'STN_D': ['BLOCK_D1'],
    'BLOCK_D1': ['STN_D', 'BLOCK_D2'],
    'BLOCK_D2': ['BLOCK_D1', 'BLOCK_D3', 'BLOCK_V_D2_A2', 'STN_E'],
    'STN_E': ['BLOCK_D2'],
    'BLOCK_D3': ['BLOCK_D2', 'BLOCK_D4'],
    'BLOCK_D4': ['BLOCK_D3', 'BLOCK_D5'],
    'BLOCK_D5': ['BLOCK_D4', 'BLOCK_B1'],
    'BLOCK_V_D2_A2': ['BLOCK_D2', 'BLOCK_A2'],
    
    # Lower branch network
    'BLOCK_F1': ['BLOCK_A1', 'BLOCK_F2'],
    'BLOCK_F2': ['BLOCK_F1', 'STN_F'],
    'STN_F': ['BLOCK_F2'],
}

# Predefined optimal routes to Station C
OPTIMAL_ROUTES = {
    'A_to_C_direct': ['STN_A', 'BLOCK_A1', 'BLOCK_A2', 'STN_B', 'BLOCK_B1', 'BLOCK_B2', 'STN_C'],
    'D_to_C_via_upper': ['STN_D', 'BLOCK_D1', 'BLOCK_D2', 'BLOCK_D3', 'BLOCK_D4', 'BLOCK_D5', 'BLOCK_B1', 'BLOCK_B2', 'STN_C'],
    'D_to_C_via_shortcut': ['STN_D', 'BLOCK_D1', 'BLOCK_D2', 'BLOCK_V_D2_A2', 'BLOCK_A2', 'STN_B', 'BLOCK_B1', 'BLOCK_B2', 'STN_C'],
}

class OptimizedTrafficControlSystem:
    def __init__(self):
        self.trains = {}
        self.block_occupancy = {}
        self.station_platforms = {}
        self.simulation_time = 0
        self.is_running = False
        self.train_progress = {}
        self.websocket_connections: Set[WebSocket] = set()
        self.scheduled_departures = {}
        self.path_optimizer = PathOptimizer()
        
        # Initialize infrastructure
        self._initialize_infrastructure()
        
    def _initialize_infrastructure(self):
        """Initialize block occupancy and station platforms"""
        for section in TRACK_SECTIONS:
            if section['type'] == 'block':
                self.block_occupancy[section['id']] = None
            elif section['type'] == 'station':
                self.station_platforms[section['id']] = {}
                for i in range(1, section['platforms'] + 1):
                    self.station_platforms[section['id']][i] = None

    async def add_websocket(self, websocket: WebSocket):
        self.websocket_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.websocket_connections)}")

    async def remove_websocket(self, websocket: WebSocket):
        self.websocket_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.websocket_connections)}")

    async def broadcast_state(self):
        if not self.websocket_connections:
            return
            
        state = {
            "trains": list(self.trains.values()),
            "blockOccupancy": self.block_occupancy,
            "stationPlatforms": self.station_platforms,
            "simulationTime": self.simulation_time,
            "isRunning": self.is_running,
            "trainProgress": self.train_progress,
            "scheduledDepartures": self.scheduled_departures
        }
        
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(state))
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.add(websocket)
        
        for ws in disconnected:
            self.websocket_connections.discard(ws)

    def is_section_available(self, section_id: str, train_id: str) -> bool:
        """Check if a section is available for occupation"""
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section:
            return False
        
        if section['type'] == 'block':
            return self.block_occupancy[section_id] is None or self.block_occupancy[section_id] == train_id
        elif section['type'] == 'station':
            platforms = self.station_platforms[section_id]
            return any(occupant is None or occupant == train_id for occupant in platforms.values())
        return False

    def occupy_section(self, section_id: str, train_id: str) -> bool:
        """Occupy a section with a train"""
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section:
            return False
        
        if section['type'] == 'block':
            if self.block_occupancy[section_id] is None:
                self.block_occupancy[section_id] = train_id
                return True
        elif section['type'] == 'station':
            platforms = self.station_platforms[section_id]
            for platform_num, occupant in platforms.items():
                if occupant is None:
                    platforms[platform_num] = train_id
                    return True
        return False

    def release_section(self, section_id: str, train_id: str):
        """Release a section from train occupation"""
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section:
            return
        
        if section['type'] == 'block':
            if self.block_occupancy[section_id] == train_id:
                self.block_occupancy[section_id] = None
        elif section['type'] == 'station':
            platforms = self.station_platforms[section_id]
            for platform_num, occupant in platforms.items():
                if occupant == train_id:
                    platforms[platform_num] = None

    def calculate_travel_time(self, section_id: str, train_speed: int) -> int:
        """Calculate travel time for a section based on train speed and type"""
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section:
            return 3
        
        # Base travel times (in simulation minutes)
        base_times = {
            'block': 3,    # 3 minutes for block sections
            'station': 1   # 1 minute for station sections (movement only)
        }
        
        section_type = section['type']
        base_time = base_times.get(section_type, 3)
        
        # Speed adjustment factor (higher speed = less time)
        speed_factor = max(0.6, min(1.5, 80 / train_speed))
        
        return max(1, int(base_time * speed_factor))

    def calculate_halt_time(self, section_id: str, train_type: str) -> int:
        """Calculate halt time at stations based on train type"""
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section or section['type'] != 'station':
            return 0
        
        # Halt times based on train type
        halt_times = {
            'Express': 1,      # Express trains: 1 minute halt
            'Passenger': 2,    # Passenger trains: 2 minutes halt
            'Freight': 3       # Freight trains: 3 minutes halt
        }
        
        return halt_times.get(train_type, 2)

    def add_optimized_train(self, train_data: dict):
        """Add a train with optimized path selection"""
        train_id = train_data['id']
        
        # Determine optimal route based on origin
        if train_data['origin'] == 'A':
            optimal_route = OPTIMAL_ROUTES['A_to_C_direct']
        elif train_data['origin'] == 'D':
            # Choose best route based on current traffic
            route_options = [
                OPTIMAL_ROUTES['D_to_C_via_upper'],
                OPTIMAL_ROUTES['D_to_C_via_shortcut']
            ]
            optimal_route = self.path_optimizer.select_best_route(route_options, self.block_occupancy)
        else:
            optimal_route = OPTIMAL_ROUTES['A_to_C_direct']
        
        # Create optimized schedule
        schedule = self._create_optimized_schedule(train_data, optimal_route)
        
        train = {
            'id': train_id,
            'name': train_data['name'],
            'number': train_data['number'],
            'type': train_data['type'],
            'section': optimal_route[0],
            'speed': train_data['speed'],
            'origin': train_data['origin'],
            'destination': 'C',
            'status': 'Scheduled',
            'statusType': 'scheduled',
            'delay': 0,
            'route': optimal_route,
            'departureTime': train_data['departureTime'],
            'schedule': schedule,
            'platform': None,
            'waitingForBlock': False,
            'haltTimes': self._calculate_route_halt_times(optimal_route, train_data['type'])
        }
        
        self.trains[train_id] = train
        
        # Initialize progress tracking
        self.train_progress[train_id] = {
            'currentRouteIndex': 0,
            'lastMoveTime': train_data['departureTime'],
            'isMoving': False,
            'nextScheduledTime': train_data['departureTime'],
            'waitingForSection': None,
            'haltEndTime': 0
        }
        
        # Schedule departure
        self.scheduled_departures[train_id] = train_data['departureTime']
        
        # Occupy initial section
        self.occupy_section(optimal_route[0], train_id)
        
        return train

    def _create_optimized_schedule(self, train_data: dict, route: List[str]) -> Dict[str, List[int]]:
        """Create an optimized schedule considering traffic and halt times"""
        schedule = {}
        current_time = train_data['departureTime']
        
        for i, section_id in enumerate(route):
            section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
            if not section:
                continue
            
            if i > 0:  # Skip origin
                # Add travel time from previous section
                travel_time = self.calculate_travel_time(section_id, train_data['speed'])
                current_time += travel_time
                
                # Add buffer for traffic optimization
                buffer_time = 1 if section['type'] == 'block' else 0
                current_time += buffer_time
            
            if section['type'] == 'station':
                halt_time = self.calculate_halt_time(section_id, train_data['type'])
                schedule[section_id] = [current_time, halt_time]
                current_time += halt_time
        
        return schedule

    def _calculate_route_halt_times(self, route: List[str], train_type: str) -> Dict[str, int]:
        """Pre-calculate halt times for all stations in route"""
        halt_times = {}
        for section_id in route:
            section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
            if section and section['type'] == 'station':
                halt_times[section_id] = self.calculate_halt_time(section_id, train_type)
        return halt_times

    async def update_simulation(self):
        """Main simulation update loop with collision avoidance"""
        if not self.is_running:
            return
        
        self.simulation_time += 1
        
        # Update all trains
        for train_id, train in list(self.trains.items()):
            await self._update_train_with_optimization(train_id, train)
        
        await self.broadcast_state()

    async def _update_train_with_optimization(self, train_id: str, train: dict):
        """Update train with optimization and collision avoidance"""
        progress = self.train_progress.get(train_id, {})
        
        # Check if train should start
        if self.simulation_time < train['departureTime']:
            train['status'] = f"Scheduled departure at {train['departureTime']:02d}:00"
            train['statusType'] = 'scheduled'
            return
        
        # Train is active
        train['statusType'] = 'running'
        current_route_index = progress.get('currentRouteIndex', 0)
        
        # Check if train has reached destination
        if current_route_index >= len(train['route']) - 1:
            train['status'] = 'Arrived at Station C'
            train['statusType'] = 'completed'
            train['waitingForBlock'] = False
            return
        
        current_section = train['route'][current_route_index]
        
        # Check if train is currently halting at a station
        if progress.get('haltEndTime', 0) > self.simulation_time:
            section = next((s for s in TRACK_SECTIONS if s['id'] == current_section), None)
            if section and section['type'] == 'station':
                remaining_halt = progress['haltEndTime'] - self.simulation_time
                train['status'] = f"Station halt - {remaining_halt}min remaining"
                train['waitingForBlock'] = False
                return
        
        # Try to move to next section
        if current_route_index < len(train['route']) - 1:
            next_section = train['route'][current_route_index + 1]
            
            # Check travel time requirement
            last_move_time = progress.get('lastMoveTime', train['departureTime'])
            time_in_current_section = self.simulation_time - last_move_time
            required_time = self.calculate_travel_time(current_section, train['speed'])
            
            if time_in_current_section >= required_time:
                # Check if next section is available
                if self.is_section_available(next_section, train_id):
                    # Move to next section
                    self._execute_train_movement(train_id, train, current_section, next_section, current_route_index, progress)
                else:
                    # Wait for next section
                    train['waitingForBlock'] = True
                    section_name = next((s['name'] for s in TRACK_SECTIONS if s['id'] == next_section), next_section)
                    train['status'] = f"Waiting for {section_name}"
                    progress['waitingForSection'] = next_section
            else:
                # Still traveling in current section
                remaining_time = required_time - time_in_current_section
                train['status'] = f"En route - {remaining_time}min"
                train['waitingForBlock'] = False

    def _execute_train_movement(self, train_id: str, train: dict, current_section: str, next_section: str, current_route_index: int, progress: dict):
        """Execute train movement from current section to next section"""
        # Release current section
        self.release_section(current_section, train_id)
        
        # Occupy next section
        self.occupy_section(next_section, train_id)
        
        # Update train position
        train['section'] = next_section
        train['waitingForBlock'] = False
        
        # Check if next section is a station that requires halt
        next_section_info = next((s for s in TRACK_SECTIONS if s['id'] == next_section), None)
        if next_section_info and next_section_info['type'] == 'station' and next_section != train['route'][-1]:
            # Calculate halt time
            halt_time = train['haltTimes'].get(next_section, 0)
            if halt_time > 0:
                progress['haltEndTime'] = self.simulation_time + halt_time
                train['status'] = f"Station halt - {halt_time}min"
            else:
                train['status'] = 'Passing through station'
        else:
            train['status'] = 'Running'
        
        # Update progress
        self.train_progress[train_id] = {
            **progress,
            'currentRouteIndex': current_route_index + 1,
            'lastMoveTime': self.simulation_time,
            'isMoving': True,
            'waitingForSection': None
        }
        
        # Add realistic speed variation
        speed_variation = (hash(f"{train_id}_{self.simulation_time}") % 10) - 5
        train['speed'] = max(30, min(120, train['speed'] + speed_variation))

    def start_simulation(self):
        self.is_running = True
        logger.info("Optimized simulation started")

    def pause_simulation(self):
        self.is_running = False
        logger.info("Simulation paused")

    def reset_simulation(self):
        """Reset simulation with optimized initial state"""
        self.is_running = False
        self.simulation_time = 0
        
        # Clear all occupancy
        self._initialize_infrastructure()
        
        # Reset all trains to initial positions
        for train_id, train in self.trains.items():
            initial_section = train['route'][0]
            train['section'] = initial_section
            train['status'] = 'Scheduled'
            train['statusType'] = 'scheduled'
            train['waitingForBlock'] = False
            train['platform'] = None
            train['delay'] = 0
            
            # Reset progress
            self.train_progress[train_id] = {
                'currentRouteIndex': 0,
                'lastMoveTime': train['departureTime'],
                'isMoving': False,
                'nextScheduledTime': train['departureTime'],
                'waitingForSection': None,
                'haltEndTime': 0
            }
            
            # Re-occupy initial section
            self.occupy_section(initial_section, train_id)
        
        logger.info("Simulation reset with optimization")

    def get_system_state(self):
        return {
            "trains": list(self.trains.values()),
            "blockOccupancy": self.block_occupancy,
            "stationPlatforms": self.station_platforms,
            "simulationTime": self.simulation_time,
            "isRunning": self.is_running,
            "trainProgress": self.train_progress,
            "trackSections": TRACK_SECTIONS,
            "scheduledDepartures": self.scheduled_departures,
            "optimalRoutes": OPTIMAL_ROUTES
        }

class PathOptimizer:
    """Handles path optimization and route selection"""
    
    def select_best_route(self, route_options: List[List[str]], current_occupancy: Dict[str, str]) -> List[str]:
        """Select the best route based on current traffic conditions"""
        best_route = route_options[0]
        min_congestion = float('inf')
        
        for route in route_options:
            congestion_score = self._calculate_route_congestion(route, current_occupancy)
            if congestion_score < min_congestion:
                min_congestion = congestion_score
                best_route = route
        
        return best_route
    
    def _calculate_route_congestion(self, route: List[str], occupancy: Dict[str, str]) -> int:
        """Calculate congestion score for a route"""
        congestion = 0
        for section_id in route:
            if occupancy.get(section_id) is not None:
                congestion += 10  # Heavy penalty for occupied sections
            
            # Additional penalty for sections likely to be occupied soon
            adjacent_sections = RAILWAY_NETWORK.get(section_id, [])
            for adj_section in adjacent_sections:
                if occupancy.get(adj_section) is not None:
                    congestion += 2  # Light penalty for adjacent occupied sections
        
        return congestion

# Global system instance
traffic_system = OptimizedTrafficControlSystem()

# Pydantic models
class TrainData(BaseModel):
    id: str
    name: str
    number: str
    type: str  # Express, Passenger, Freight
    origin: str  # A or D
    speed: int
    departureTime: int

class SimulationControl(BaseModel):
    action: str

# Background simulation loop
async def simulation_loop():
    while True:
        if traffic_system.is_running:
            await traffic_system.update_simulation()
        await asyncio.sleep(1.5)  # Slightly faster updates

# Startup event
@app.on_event("startup")
async def startup_event():
    # Add optimized train fleet
    optimized_trains = [
        # 3 trains starting from Station A with staggered departures
        {
            'id': 'A1', 'name': 'Mumbai Express', 'number': '12951', 'type': 'Express',
            'origin': 'A', 'speed': 85, 'departureTime': 0
        },
        {
            'id': 'A2', 'name': 'Chennai Passenger', 'number': '56041', 'type': 'Passenger',
            'origin': 'A', 'speed': 65, 'departureTime': 4
        },
        {
            'id': 'A3', 'name': 'Bangalore Express', 'number': '12639', 'type': 'Express',
            'origin': 'A', 'speed': 80, 'departureTime': 7
        },
        
        # 2 trains starting from Station D with optimized timing
        {
            'id': 'D1', 'name': 'Delhi Rajdhani', 'number': '12301', 'type': 'Express',
            'origin': 'D', 'speed': 90, 'departureTime': 2
        },
        {
            'id': 'D2', 'name': 'Kolkata Mail', 'number': '12841', 'type': 'Passenger',
            'origin': 'D', 'speed': 70, 'departureTime': 6
        }
    ]
    
    for train_data in optimized_trains:
        traffic_system.add_optimized_train(train_data)
    
    # Start simulation loop
    asyncio.create_task(simulation_loop())
    logger.info("Optimized Railway Traffic Control System initialized with 5 trains")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await traffic_system.add_websocket(websocket)
    
    try:
        await websocket.send_text(json.dumps(traffic_system.get_system_state()))
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Received websocket message: {message}")
    except WebSocketDisconnect:
        await traffic_system.remove_websocket(websocket)

# API endpoints
@app.get("/")
async def root():
    return {"message": "Optimized Railway Traffic Control Backend API v3.0", "status": "active"}

@app.get("/system-state")
async def get_system_state():
    return traffic_system.get_system_state()

@app.post("/simulation-control")
async def control_simulation(control: SimulationControl):
    if control.action == "start":
        traffic_system.start_simulation()
    elif control.action == "pause":
        traffic_system.pause_simulation()
    elif control.action == "reset":
        traffic_system.reset_simulation()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    await traffic_system.broadcast_state()
    return {"status": "success", "action": control.action}

@app.post("/add-train")
async def add_train(train: TrainData):
    try:
        new_train = traffic_system.add_optimized_train(train.dict())
        await traffic_system.broadcast_state()
        return {"status": "success", "train": new_train}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimization-stats")
async def get_optimization_stats():
    """Get optimization and performance statistics"""
    total_trains = len(traffic_system.trains)
    running_trains = sum(1 for t in traffic_system.trains.values() if t['statusType'] == 'running')
    waiting_trains = sum(1 for t in traffic_system.trains.values() if t['waitingForBlock'])
    completed_trains = sum(1 for t in traffic_system.trains.values() if t['statusType'] == 'completed')
    
    # Route distribution
    route_usage = {}
    for train in traffic_system.trains.values():
        route_key = f"{train['origin']}_to_C"
        route_usage[route_key] = route_usage.get(route_key, 0) + 1
    
    # Infrastructure utilization
    occupied_blocks = sum(1 for occupant in traffic_system.block_occupancy.values() if occupant is not None)
    total_blocks = len(traffic_system.block_occupancy)
    
    occupied_platforms = 0
    total_platforms = 0
    for station_platforms in traffic_system.station_platforms.values():
        total_platforms += len(station_platforms)
        occupied_platforms += sum(1 for occupant in station_platforms.values() if occupant is not None)
    
    return {
        "trains": {
            "total": total_trains,
            "running": running_trains,
            "waiting": waiting_trains,
            "completed": completed_trains,
            "throughput_efficiency": round((completed_trains / max(total_trains, 1)) * 100, 1)
        },
        "routes": {
            "usage": route_usage,
            "optimization_active": True
        },
        "infrastructure": {
            "blocks": {
                "total": total_blocks,
                "occupied": occupied_blocks,
                "free": total_blocks - occupied_blocks,
                "utilization": round((occupied_blocks / max(total_blocks, 1)) * 100, 1)
            },
            "platforms": {
                "total": total_platforms,
                "occupied": occupied_platforms,
                "free": total_platforms - occupied_platforms,
                "utilization": round((occupied_platforms / max(total_platforms, 1)) * 100, 1)
            }
        },
        "simulation": {
            "time": traffic_system.simulation_time,
            "running": traffic_system.is_running,
            "optimization_enabled": True
        }
    }

@app.get("/routes")
async def get_routes():
    """Get available optimized routes"""
    return {"routes": OPTIMAL_ROUTES, "network": RAILWAY_NETWORK}

@app.get("/track-sections")
async def get_track_sections():
    """Get all track sections information"""
    return {"sections": TRACK_SECTIONS}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")