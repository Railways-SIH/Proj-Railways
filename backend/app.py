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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Railway Traffic Control Backend", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track sections mapping to match frontend
TRACK_SECTIONS = [
    {'id': 'ENTRY_BLOCK', 'type': 'block', 'name': 'Entry Block', 'station': None},
    {'id': 'STN_A', 'type': 'station', 'name': 'Central Stn', 'station': 'A', 'platforms': 3},
    {'id': 'STN_B', 'type': 'station', 'name': 'Junction Stn', 'station': 'B', 'platforms': 2},
    {'id': 'BLOCK_AB', 'type': 'block', 'name': 'AB Block', 'station': None},
    {'id': 'BLOCK_BC', 'type': 'block', 'name': 'BC Block', 'station': None},
    {'id': 'STN_C', 'type': 'station', 'name': 'Metro Stn', 'station': 'C', 'platforms': 2},
    {'id': 'BLOCK_CD1', 'type': 'block', 'name': 'CD Block 1', 'station': None},
    {'id': 'BLOCK_CD2', 'type': 'block', 'name': 'CD Block 2', 'station': None},
    {'id': 'STN_D', 'type': 'station', 'name': 'Terminal Stn', 'station': 'D', 'platforms': 4},
    # Branch lines
    {'id': 'BRANCH_N1', 'type': 'block', 'name': 'North Branch 1', 'station': None},
    {'id': 'BRANCH_N2', 'type': 'block', 'name': 'North Branch 2', 'station': None},
    {'id': 'BRANCH_N3', 'type': 'block', 'name': 'North Branch 3', 'station': None},
    {'id': 'BRANCH_N4', 'type': 'block', 'name': 'North Branch 4', 'station': None},
    {'id': 'BRANCH_S1', 'type': 'block', 'name': 'South Branch 1', 'station': None},
    {'id': 'BRANCH_S2', 'type': 'block', 'name': 'South Branch 2', 'station': None},
    {'id': 'BRANCH_S3', 'type': 'block', 'name': 'South Branch 3', 'station': None},
    {'id': 'BRANCH_S4', 'type': 'block', 'name': 'South Branch 4', 'station': None},
    {'id': 'YARD_1', 'type': 'block', 'name': 'Yard Block 1', 'station': None},
    {'id': 'YARD_2', 'type': 'block', 'name': 'Yard Block 2', 'station': None},
    {'id': 'YARD_3', 'type': 'block', 'name': 'Yard Block 3', 'station': None},
    {'id': 'YARD_4', 'type': 'block', 'name': 'Yard Block 4', 'station': None},
]

# Route definitions
MAIN_ROUTES = {
    'A_to_D': ['STN_A', 'STN_B', 'BLOCK_AB', 'BLOCK_BC', 'STN_C', 'BLOCK_CD1', 'BLOCK_CD2', 'STN_D'],
    'B_to_D': ['STN_B', 'BLOCK_AB', 'BLOCK_BC', 'STN_C', 'BLOCK_CD1', 'BLOCK_CD2', 'STN_D'],
    'C_to_D': ['STN_C', 'BLOCK_CD1', 'BLOCK_CD2', 'STN_D']
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
        
        # Initialize block occupancy
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
            "trainProgress": self.train_progress
        }
        
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(state))
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.websocket_connections.discard(ws)

    def is_section_available(self, section_id: str, train_id: str) -> bool:
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section:
            return False
        
        if section['type'] == 'block':
            return self.block_occupancy[section_id] is None or self.block_occupancy[section_id] == train_id
        elif section['type'] == 'station':
            platforms = self.station_platforms[section_id]
            return any(occupant is None or occupant == train_id for occupant in platforms.values())
        return False

    def occupy_section(self, section_id: str, train_id: str):
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section:
            return False
        
        if section['type'] == 'block':
            self.block_occupancy[section_id] = train_id
            return True
        elif section['type'] == 'station':
            platforms = self.station_platforms[section_id]
            for platform_num, occupant in platforms.items():
                if occupant is None or occupant == train_id:
                    platforms[platform_num] = train_id
                    return True
        return False

    def release_section(self, section_id: str, train_id: str):
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
        """Calculate travel time for a section based on train speed"""
        base_times = {
            'block': 4,  # 4 minutes base for block sections
            'station': 2  # 2 minutes base for station sections
        }
        
        section = next((s for s in TRACK_SECTIONS if s['id'] == section_id), None)
        if not section:
            return 4
        
        section_type = section['type']
        base_time = base_times.get(section_type, 4)
        
        # Adjust based on speed (faster trains take less time)
        speed_factor = max(0.5, min(2.0, 80 / train_speed))
        return max(2, int(base_time * speed_factor))

    def add_train(self, train_data: dict):
        """Add a train with optimized scheduling"""
        train_id = train_data['id']
        
        # Generate route based on start and destination
        route_key = f"{train_data['start']}_{train_data['destination']}"
        route = MAIN_ROUTES.get(route_key, MAIN_ROUTES['A_to_D'])
        
        train = {
            'id': train_id,
            'name': train_data['name'],
            'number': train_data['number'],
            'section': route[0],
            'speed': train_data['speed'],
            'destination': train_data['destination'],
            'status': 'Scheduled',
            'statusType': 'scheduled',
            'delay': 0,
            'route': route,
            'departureTime': train_data.get('departureTime', 0),
            'schedule': self._optimize_schedule(train_data, route),
            'platform': None,
            'waitingForBlock': False
        }
        
        self.trains[train_id] = train
        
        # Initialize progress tracking
        self.train_progress[train_id] = {
            'currentRouteIndex': 0,
            'lastMoveTime': 0,
            'isMoving': False,
            'nextScheduledTime': 0,
            'waitingForSection': None
        }
        
        # Occupy initial section
        self.occupy_section(route[0], train_id)
        
        return train

    def _optimize_schedule(self, train_data: dict, route: List[str]) -> Dict[str, List[int]]:
        """Generate optimized schedule for a train considering other trains"""
        schedule = {}
        current_time = train_data.get('departureTime', 0)
        
        # Only schedule station stops
        station_sections = [s for s in route if s.startswith('STN_')]
        
        for i, section_id in enumerate(station_sections):
            if i == 0:  # Skip origin station
                continue
                
            # Calculate travel time from previous station
            sections_to_traverse = []
            prev_station_idx = route.index(station_sections[i-1]) if i > 0 else 0
            current_station_idx = route.index(section_id)
            
            # Add travel time for all sections between stations
            travel_time = 0
            for j in range(prev_station_idx + 1, current_station_idx + 1):
                travel_time += self.calculate_travel_time(route[j], train_data['speed'])
            
            current_time += travel_time
            
            # Add buffer time to avoid conflicts
            current_time += 1
            
            # Determine platform (simple assignment for now)
            platform = 1 if i % 2 == 1 else 2
            
            # Station dwell time
            dwell_time = 2 if 'Express' in train_data['name'] else 1
            
            schedule[section_id] = [current_time, dwell_time]
            current_time += dwell_time
        
        return schedule

    async def update_simulation(self):
        """Update train positions and states"""
        if not self.is_running:
            return
        
        self.simulation_time += 1
        
        for train_id, train in self.trains.items():
            await self._update_train(train_id, train)
        
        await self.broadcast_state()

    async def _update_train(self, train_id: str, train: dict):
        """Update individual train state"""
        progress = self.train_progress.get(train_id, {})
        
        # Check if train should start moving
        if self.simulation_time < train['departureTime']:
            train['status'] = f"Departing at {train['departureTime']:02d}:00"
            train['statusType'] = 'scheduled'
            return
        
        # Train is now active
        train['statusType'] = 'running'
        current_route_index = progress.get('currentRouteIndex', 0)
        
        if current_route_index >= len(train['route']) - 1:
            # Train has reached destination
            train['status'] = 'Arrived at Terminal Station'
            train['statusType'] = 'completed'
            train['waitingForBlock'] = False
            return
        
        current_section = train['route'][current_route_index]
        next_section = train['route'][current_route_index + 1]
        
        # Check if it's time to move to next section
        last_move_time = progress.get('lastMoveTime', train['departureTime'])
        time_in_current_section = self.simulation_time - last_move_time
        
        # Calculate required time in current section
        required_time = self.calculate_travel_time(current_section, train['speed'])
        
        # Check scheduled stops
        if current_section in train['schedule']:
            scheduled_arrival, dwell_time = train['schedule'][current_section]
            if self.simulation_time < scheduled_arrival + dwell_time:
                train['status'] = f"Scheduled stop at {current_section}"
                train['waitingForBlock'] = False
                return
        
        # Try to move if enough time has passed
        if time_in_current_section >= required_time:
            if self.is_section_available(next_section, train_id):
                # Move to next section
                self.release_section(current_section, train_id)
                self.occupy_section(next_section, train_id)
                
                # Update train state
                train['section'] = next_section
                train['status'] = 'Running'
                train['waitingForBlock'] = False
                
                # Update progress
                self.train_progress[train_id] = {
                    **progress,
                    'currentRouteIndex': current_route_index + 1,
                    'lastMoveTime': self.simulation_time,
                    'isMoving': True,
                    'waitingForSection': None
                }
                
                # Add speed variation
                speed_variation = (hash(f"{train_id}_{self.simulation_time}") % 16) - 8
                train['speed'] = max(25, min(120, train['speed'] + speed_variation))
                
            else:
                # Waiting for next section
                train['waitingForBlock'] = True
                section_name = next((s['name'] for s in TRACK_SECTIONS if s['id'] == next_section), next_section)
                train['status'] = f"Waiting for {section_name}"
                
                self.train_progress[train_id] = {
                    **progress,
                    'waitingForSection': next_section
                }

    def start_simulation(self):
        """Start the simulation"""
        self.is_running = True
        logger.info("Simulation started")

    def pause_simulation(self):
        """Pause the simulation"""
        self.is_running = False
        logger.info("Simulation paused")

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.is_running = False
        self.simulation_time = 0
        
        # Clear occupancy
        for section in TRACK_SECTIONS:
            if section['type'] == 'block':
                self.block_occupancy[section['id']] = None
            elif section['type'] == 'station':
                for platform in self.station_platforms[section['id']]:
                    self.station_platforms[section['id']][platform] = None
        
        # Reset trains to initial positions
        for train_id, train in self.trains.items():
            initial_section = train['route'][0]
            train['section'] = initial_section
            train['status'] = 'Scheduled'
            train['statusType'] = 'scheduled'
            train['waitingForBlock'] = False
            train['platform'] = None
            
            # Reset progress
            self.train_progress[train_id] = {
                'currentRouteIndex': 0,
                'lastMoveTime': 0,
                'isMoving': False,
                'nextScheduledTime': 0,
                'waitingForSection': None
            }
            
            # Occupy initial section
            self.occupy_section(initial_section, train_id)
        
        logger.info("Simulation reset")

    def get_system_state(self):
        """Get complete system state"""
        return {
            "trains": list(self.trains.values()),
            "blockOccupancy": self.block_occupancy,
            "stationPlatforms": self.station_platforms,
            "simulationTime": self.simulation_time,
            "isRunning": self.is_running,
            "trainProgress": self.train_progress,
            "trackSections": TRACK_SECTIONS
        }

# Global system instance
traffic_system = TrafficControlSystem()

# Pydantic models
class TrainData(BaseModel):
    id: str
    name: str
    number: str
    start: str
    destination: str
    speed: int
    departureTime: int = 0

class SimulationControl(BaseModel):
    action: str  # 'start', 'pause', 'reset'

# Background task for simulation updates
async def simulation_loop():
    """Background task that runs the simulation"""
    while True:
        if traffic_system.is_running:
            await traffic_system.update_simulation()
        await asyncio.sleep(1.8)  # Match frontend timing

# Start background task
@app.on_event("startup")
async def startup_event():
    # Add default trains
    default_trains = [
        {
            'id': 'T1',
            'name': 'Rajdhani Express',
            'number': '12301',
            'start': 'A',
            'destination': 'D',
            'speed': 80,
            'departureTime': 0
        },
        {
            'id': 'T2',
            'name': 'Shatabdi Express',
            'number': '12002',
            'start': 'A',
            'destination': 'D',
            'speed': 60,
            'departureTime': 3
        },
        {
            'id': 'T3',
            'name': 'Duronto Express',
            'number': '12259',
            'start': 'A',
            'destination': 'D',
            'speed': 45,
            'departureTime': 6
        }
    ]
    
    for train_data in default_trains:
        traffic_system.add_train(train_data)
    
    # Start simulation loop
    asyncio.create_task(simulation_loop())
    logger.info("Railway Traffic Control System initialized")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await traffic_system.add_websocket(websocket)
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps(traffic_system.get_system_state()))
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client messages if needed
            logger.info(f"Received websocket message: {message}")
            
    except WebSocketDisconnect:
        await traffic_system.remove_websocket(websocket)

# API endpoints
@app.get("/")
async def root():
    return {"message": "Railway Traffic Control Backend API v2.0", "status": "active"}

@app.get("/system-state")
async def get_system_state():
    """Get complete system state"""
    return traffic_system.get_system_state()

@app.post("/simulation-control")
async def control_simulation(control: SimulationControl):
    """Control simulation (start/pause/reset)"""
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
    """Add a new train to the system"""
    try:
        new_train = traffic_system.add_train(train.dict())
        await traffic_system.broadcast_state()
        return {"status": "success", "train": new_train}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/track-sections")
async def get_track_sections():
    """Get all track sections information"""
    return {"sections": TRACK_SECTIONS}

@app.get("/routes")
async def get_routes():
    """Get available routes"""
    return {"routes": MAIN_ROUTES}

@app.get("/performance-stats")
async def get_performance_stats():
    """Get system performance statistics"""
    total_trains = len(traffic_system.trains)
    running_trains = sum(1 for t in traffic_system.trains.values() if t['statusType'] == 'running')
    waiting_trains = sum(1 for t in traffic_system.trains.values() if t['waitingForBlock'])
    completed_trains = sum(1 for t in traffic_system.trains.values() if t['statusType'] == 'completed')
    
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
            "completed": completed_trains
        },
        "infrastructure": {
            "blocks": {
                "total": total_blocks,
                "occupied": occupied_blocks,
                "free": total_blocks - occupied_blocks
            },
            "platforms": {
                "total": total_platforms,
                "occupied": occupied_platforms,
                "free": total_platforms - occupied_platforms
            }
        },
        "simulation": {
            "time": traffic_system.simulation_time,
            "running": traffic_system.is_running
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")