from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Set, Tuple, Optional
import heapq
import uvicorn
import asyncio
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import uuid
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Enhanced Intelligent Railway Control Backend", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expanded track sections with more stations and blocks
TRACK_SECTIONS = [
    # Main line stations and blocks
    {'id': 'STN_A', 'type': 'station', 'name': 'Central Station A', 'station': 'A', 'platforms': 4}, 
    {'id': 'BLOCK_A1', 'type': 'block', 'name': 'Block A1'},
    {'id': 'BLOCK_A2', 'type': 'block', 'name': 'Block A2'}, 
    {'id': 'STN_B', 'type': 'junction', 'name': 'Junction B', 'station': 'B', 'platforms': 3},
    {'id': 'BLOCK_B1', 'type': 'block', 'name': 'Block B1'}, 
    {'id': 'BLOCK_B2', 'type': 'block', 'name': 'Block B2'},
    {'id': 'STN_C', 'type': 'junction', 'name': 'Metro C', 'station': 'C', 'platforms': 3}, 
    {'id': 'BLOCK_C1', 'type': 'block', 'name': 'Block C1'},
    {'id': 'BLOCK_C2', 'type': 'block', 'name': 'Block C2'}, 
    {'id': 'STN_D', 'type': 'station', 'name': 'Terminal D', 'station': 'D', 'platforms': 2},
    
    # Northern branch
    {'id': 'STN_E', 'type': 'station', 'name': 'North Hub E', 'station': 'E', 'platforms': 3}, 
    {'id': 'BLOCK_E1', 'type': 'block', 'name': 'Block E1'},
    {'id': 'BLOCK_E2', 'type': 'block', 'name': 'Block E2'}, 
    {'id': 'STN_F', 'type': 'junction', 'name': 'Express F', 'station': 'F', 'platforms': 2},
    {'id': 'BLOCK_F1', 'type': 'block', 'name': 'Block F1'}, 
    {'id': 'BLOCK_F2', 'type': 'block', 'name': 'Block F2'},
    {'id': 'STN_G', 'type': 'station', 'name': 'Regional G', 'station': 'G', 'platforms': 2},
    
    # Upper branch
    {'id': 'STN_H', 'type': 'station', 'name': 'Summit H', 'station': 'H', 'platforms': 2}, 
    {'id': 'BLOCK_H1', 'type': 'block', 'name': 'Block H1'},
    {'id': 'BLOCK_H2', 'type': 'block', 'name': 'Block H2'}, 
    {'id': 'STN_I', 'type': 'station', 'name': 'Peak I', 'station': 'I', 'platforms': 2},
    
    # Southern branch
    {'id': 'STN_J', 'type': 'station', 'name': 'South Bay J', 'station': 'J', 'platforms': 3}, 
    {'id': 'BLOCK_J1', 'type': 'block', 'name': 'Block J1'},
    {'id': 'BLOCK_J2', 'type': 'block', 'name': 'Block J2'}, 
    {'id': 'STN_K', 'type': 'junction', 'name': 'Coast K', 'station': 'K', 'platforms': 2},
    {'id': 'BLOCK_K1', 'type': 'block', 'name': 'Block K1'}, 
    {'id': 'STN_L', 'type': 'station', 'name': 'Harbor L', 'station': 'L', 'platforms': 2},
    
    # Junction blocks
    {'id': 'BLOCK_V_A_E', 'type': 'block', 'name': 'V-Block (A-E)'},
    {'id': 'BLOCK_V_A_J', 'type': 'block', 'name': 'V-Block (A-J)'},
    {'id': 'BLOCK_V_B_F', 'type': 'block', 'name': 'V-Block (B-F)'},
    {'id': 'BLOCK_V_F_H', 'type': 'block', 'name': 'V-Block (F-H)'},
    {'id': 'BLOCK_V_B_K', 'type': 'block', 'name': 'V-Block (B-K)'},
    {'id': 'BLOCK_V_C_G', 'type': 'block', 'name': 'V-Block (C-G)'},
]

# Enhanced graph with multiple paths for deadlock prevention
GRAPH = {
    # Main line connections
    'STN_A': {'BLOCK_A1': 5, 'BLOCK_V_A_E': 4, 'BLOCK_V_A_J': 4}, 
    'BLOCK_A1': {'STN_A': 5, 'BLOCK_A2': 5},
    'BLOCK_A2': {'BLOCK_A1': 5, 'STN_B': 5}, 
    'STN_B': {'BLOCK_A2': 5, 'BLOCK_B1': 5, 'BLOCK_V_B_F': 4, 'BLOCK_V_B_K': 4},
    'BLOCK_B1': {'STN_B': 5, 'BLOCK_B2': 5}, 
    'BLOCK_B2': {'BLOCK_B1': 5, 'STN_C': 5},
    'STN_C': {'BLOCK_B2': 5, 'BLOCK_C1': 5, 'BLOCK_V_C_G': 4}, 
    'BLOCK_C1': {'STN_C': 5, 'BLOCK_C2': 5},
    'BLOCK_C2': {'BLOCK_C1': 5, 'STN_D': 5}, 
    'STN_D': {'BLOCK_C2': 5},
    
    # Northern branch connections
    'STN_E': {'BLOCK_E1': 5, 'BLOCK_V_A_E': 4}, 
    'BLOCK_E1': {'STN_E': 5, 'BLOCK_E2': 5},
    'BLOCK_E2': {'BLOCK_E1': 5, 'STN_F': 5}, 
    'STN_F': {'BLOCK_E2': 5, 'BLOCK_F1': 5, 'BLOCK_V_B_F': 4, 'BLOCK_V_F_H': 4},
    'BLOCK_F1': {'STN_F': 5, 'BLOCK_F2': 5}, 
    'BLOCK_F2': {'BLOCK_F1': 5, 'STN_G': 5},
    'STN_G': {'BLOCK_F2': 5, 'BLOCK_V_C_G': 4},
    
    # Upper branch connections
    'STN_H': {'BLOCK_H1': 5, 'BLOCK_V_F_H': 4}, 
    'BLOCK_H1': {'STN_H': 5, 'BLOCK_H2': 5},
    'BLOCK_H2': {'BLOCK_H1': 5, 'STN_I': 5}, 
    'STN_I': {'BLOCK_H2': 5},
    
    # Southern branch connections
    'STN_J': {'BLOCK_J1': 5, 'BLOCK_V_A_J': 4}, 
    'BLOCK_J1': {'STN_J': 5, 'BLOCK_J2': 5},
    'BLOCK_J2': {'BLOCK_J1': 5, 'STN_K': 5}, 
    'STN_K': {'BLOCK_J2': 5, 'BLOCK_K1': 5, 'BLOCK_V_B_K': 4},
    'BLOCK_K1': {'STN_K': 5, 'STN_L': 5}, 
    'STN_L': {'BLOCK_K1': 5},
    
    # Junction block connections
    'BLOCK_V_A_E': {'STN_A': 4, 'STN_E': 4},
    'BLOCK_V_A_J': {'STN_A': 4, 'STN_J': 4},
    'BLOCK_V_B_F': {'STN_B': 4, 'STN_F': 4},
    'BLOCK_V_F_H': {'STN_F': 4, 'STN_H': 4},
    'BLOCK_V_B_K': {'STN_B': 4, 'STN_K': 4},
    'BLOCK_V_C_G': {'STN_C': 4, 'STN_G': 4},
}

@dataclass
class HistoricalRecord:
    train_id: str
    route_length: int
    scheduled_speed: int
    actual_speed: int
    weather_condition: str
    time_of_day: int
    network_congestion: float
    actual_travel_time: int
    delay: int
    timestamp: datetime

@dataclass
class OptimizationRecommendation:
    id: str
    type: str
    train_id: str
    train_number: str
    priority: int
    reason: str
    current_value: any
    recommended_value: any
    confidence: float
    created_at: datetime

class DeadlockDetector:
    def __init__(self):
        self.waiting_graph = defaultdict(set)
        self.deadlock_count = 0
        
    def add_waiting_relation(self, waiting_train: str, blocking_train: str):
        """Add a waiting relationship between trains"""
        self.waiting_graph[waiting_train].add(blocking_train)
        
    def remove_waiting_relation(self, waiting_train: str, blocking_train: str = None):
        """Remove waiting relationship"""
        if blocking_train:
            self.waiting_graph[waiting_train].discard(blocking_train)
        else:
            self.waiting_graph[waiting_train].clear()
            
    def detect_deadlock(self) -> List[List[str]]:
        """Detect circular waiting patterns (deadlocks)"""
        visited = set()
        rec_stack = set()
        deadlock_cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                if len(cycle) > 1:
                    deadlock_cycles.append(cycle)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.waiting_graph.get(node, []):
                dfs(neighbor, path)
                
            rec_stack.remove(node)
            path.pop()
            
        for train in self.waiting_graph:
            if train not in visited:
                dfs(train, [])
                
        return deadlock_cycles
        
    def clear_waiting_graph(self):
        """Clear all waiting relationships"""
        self.waiting_graph.clear()

class ConflictResolver:
    def __init__(self, traffic_system):
        self.traffic_system = traffic_system
        self.deadlock_detector = DeadlockDetector()
        
    def resolve_conflicts(self):
        """Main conflict resolution method"""
        # Update waiting relationships
        self.deadlock_detector.clear_waiting_graph()
        
        # Build current waiting graph
        for train_id, train in self.traffic_system.trains.items():
            if train['waitingForBlock']:
                blocking_trains = self._find_blocking_trains(train_id, train)
                for blocking_train in blocking_trains:
                    self.deadlock_detector.add_waiting_relation(train_id, blocking_train)
        
        # Detect deadlocks
        deadlock_cycles = self.deadlock_detector.detect_deadlock()
        
        if deadlock_cycles:
            logger.warning(f"Deadlock detected involving {len(deadlock_cycles)} cycles")
            self._resolve_deadlocks(deadlock_cycles)
            
        # Apply priority-based conflict resolution
        self._resolve_priority_conflicts()
        
    def _find_blocking_trains(self, train_id: str, train: dict) -> List[str]:
        """Find trains that are blocking this train"""
        blocking_trains = []
        progress = self.traffic_system.train_progress.get(train_id, {})
        current_idx = progress.get('currentRouteIndex', 0)
        
        if current_idx < len(train['route']) - 1:
            next_section = train['route'][current_idx + 1]
            
            # Check block occupancy
            if next_section in self.traffic_system.block_occupancy:
                occupant = self.traffic_system.block_occupancy[next_section]
                if occupant and occupant != train_id:
                    blocking_trains.append(occupant)
            
            # Check station platforms
            elif next_section in self.traffic_system.station_platforms:
                for platform, occupant in self.traffic_system.station_platforms[next_section].items():
                    if occupant and occupant != train_id:
                        blocking_trains.append(occupant)
                        break  # Only need one blocking train per station
                        
        return blocking_trains
        
    def _resolve_deadlocks(self, deadlock_cycles: List[List[str]]):
        """Resolve detected deadlocks by rerouting or priority adjustment"""
        for cycle in deadlock_cycles:
            self.deadlock_detector.deadlock_count += 1
            
            # Strategy 1: Find alternative routes for lowest priority trains
            lowest_priority_train = min(cycle, 
                key=lambda t: self.traffic_system.trains.get(t, {}).get('priority', 99))
            
            if self._attempt_reroute(lowest_priority_train):
                self.traffic_system.events.append(
                    f"Deadlock Resolved: Rerouted train {self.traffic_system.trains[lowest_priority_train]['number']}")
                continue
                
            # Strategy 2: Temporarily boost priority of one train to break cycle
            highest_priority_train = min(cycle,
                key=lambda t: self.traffic_system.trains.get(t, {}).get('priority', 99))
            
            old_priority = self.traffic_system.trains[highest_priority_train].get('priority', 99)
            self.traffic_system.trains[highest_priority_train]['priority'] = 1
            
            self.traffic_system.events.append(
                f"Deadlock Resolved: Boosted priority of train {self.traffic_system.trains[highest_priority_train]['number']}")
                
    def _attempt_reroute(self, train_id: str) -> bool:
        """Attempt to find an alternative route for a train"""
        train = self.traffic_system.trains.get(train_id)
        if not train:
            return False
            
        progress = self.traffic_system.train_progress.get(train_id, {})
        current_idx = progress.get('currentRouteIndex', 0)
        current_section = train['route'][current_idx]
        
        # Find destination
        destination = train['route'][-1]
        
        # Try to find alternative path from current position
        alternative_paths = self.traffic_system.find_all_paths(current_section, destination, max_paths=3)
        
        for alt_path in alternative_paths:
            if alt_path != train['route'][current_idx:] and len(alt_path) <= len(train['route']) * 1.5:
                # Check if alternative path is less congested
                congestion_score = self._calculate_path_congestion(alt_path)
                if congestion_score < 0.7:  # Less than 70% congested
                    # Update train route
                    new_route = train['route'][:current_idx] + alt_path
                    train['route'] = new_route
                    train['waitingForBlock'] = False
                    return True
                    
        return False
        
    def _calculate_path_congestion(self, path: List[str]) -> float:
        """Calculate congestion score for a path (0 = empty, 1 = fully occupied)"""
        occupied_sections = 0
        total_sections = len(path)
        
        for section in path:
            if section in self.traffic_system.block_occupancy:
                if self.traffic_system.block_occupancy[section] is not None:
                    occupied_sections += 1
            elif section in self.traffic_system.station_platforms:
                platforms = self.traffic_system.station_platforms[section]
                occupied_platforms = sum(1 for p in platforms.values() if p is not None)
                if occupied_platforms == len(platforms):  # Fully occupied station
                    occupied_sections += 1
                elif occupied_platforms > 0:  # Partially occupied
                    occupied_sections += 0.5
                    
        return occupied_sections / total_sections if total_sections > 0 else 0
        
    def _resolve_priority_conflicts(self):
        """Resolve conflicts based on train priorities"""
        waiting_trains = [(tid, t) for tid, t in self.traffic_system.trains.items() if t['waitingForBlock']]
        
        # Sort by priority (lower number = higher priority)
        waiting_trains.sort(key=lambda x: x[1].get('priority', 99))
        
        for train_id, train in waiting_trains:
            progress = self.traffic_system.train_progress.get(train_id, {})
            current_idx = progress.get('currentRouteIndex', 0)
            
            if current_idx < len(train['route']) - 1:
                next_section = train['route'][current_idx + 1]
                
                # Try to preempt lower priority trains
                if self._can_preempt_section(train_id, next_section):
                    self._preempt_section(train_id, next_section)

    def _can_preempt_section(self, train_id: str, section_id: str) -> bool:
        """Check if a train can preempt a section based on priority"""
        requesting_priority = self.traffic_system.trains[train_id].get('priority', 99)
        
        if section_id in self.traffic_system.block_occupancy:
            occupant_id = self.traffic_system.block_occupancy[section_id]
            if occupant_id:
                occupant_priority = self.traffic_system.trains.get(occupant_id, {}).get('priority', 99)
                return requesting_priority < occupant_priority - 5  # Significant priority difference
                
        elif section_id in self.traffic_system.station_platforms:
            platforms = self.traffic_system.station_platforms[section_id]
            for platform, occupant_id in platforms.items():
                if occupant_id:
                    occupant_priority = self.traffic_system.trains.get(occupant_id, {}).get('priority', 99)
                    if requesting_priority < occupant_priority - 5:
                        return True
                        
        return False
        
    def _preempt_section(self, train_id: str, section_id: str):
        """Preempt a section by moving lower priority train"""
        train = self.traffic_system.trains[train_id]
        
        if section_id in self.traffic_system.block_occupancy:
            occupant_id = self.traffic_system.block_occupancy[section_id]
            if occupant_id:
                occupant_train = self.traffic_system.trains[occupant_id]
                # Force the occupant to move or wait elsewhere
                self._force_train_movement(occupant_id)
                
        elif section_id in self.traffic_system.station_platforms:
            platforms = self.traffic_system.station_platforms[section_id]
            for platform, occupant_id in platforms.items():
                if occupant_id:
                    occupant_train = self.traffic_system.trains[occupant_id]
                    if train.get('priority', 99) < occupant_train.get('priority', 99) - 5:
                        self._force_train_movement(occupant_id)
                        break
                        
    def _force_train_movement(self, train_id: str):
        """Force a train to move to resolve conflicts"""
        train = self.traffic_system.trains.get(train_id)
        if not train:
            return
            
        # Try to find an alternative position or route
        progress = self.traffic_system.train_progress.get(train_id, {})
        current_idx = progress.get('currentRouteIndex', 0)
        
        # Speed up the train to clear the section faster
        train['speed'] = min(160, int(train['speed'] * 1.2))
        train['priority'] = min(1, train.get('priority', 99) - 10)
        
        self.traffic_system.events.append(
            f"Priority Override: Train {train['number']} expedited to resolve conflict")

class SyntheticDataGenerator:
    def __init__(self):
        self.weather_conditions = ['clear', 'rain', 'fog', 'snow', 'storm']
        self.historical_data = []
        
    def generate_historical_data(self, num_records=2000):
        """Generate synthetic historical train data for ML training"""
        data = []
        
        for i in range(num_records):
            route_length = random.randint(3, 15)
            scheduled_speed = random.choice([40, 60, 80, 100, 120, 140, 160])
            weather = random.choice(self.weather_conditions)
            time_of_day = random.randint(0, 23)
            network_congestion = random.uniform(0.1, 0.9)
            
            base_travel_time = route_length * 5
            weather_factor = {'clear': 1.0, 'rain': 1.1, 'fog': 1.3, 'snow': 1.4, 'storm': 1.6}[weather]
            speed_factor = max(0.7, min(1.5, 100 / scheduled_speed))
            congestion_factor = 1.0 + (network_congestion * 0.5)
            time_factor = 1.0 + (0.2 if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19 else 0)
            
            actual_travel_time = int(base_travel_time * weather_factor * speed_factor * congestion_factor * time_factor)
            delay = max(0, actual_travel_time - base_travel_time)
            
            record = HistoricalRecord(
                train_id=f"T{i}",
                route_length=route_length,
                scheduled_speed=scheduled_speed,
                actual_speed=int(scheduled_speed * random.uniform(0.8, 1.0)),
                weather_condition=weather,
                time_of_day=time_of_day,
                network_congestion=network_congestion,
                actual_travel_time=actual_travel_time,
                delay=delay,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 365))
            )
            data.append(record)
            
        self.historical_data = data
        return data

class MLETAPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, data):
        """Convert historical data to ML features"""
        df = pd.DataFrame([{
            'route_length': r.route_length,
            'scheduled_speed': r.scheduled_speed,
            'actual_speed': r.actual_speed,
            'weather_clear': 1 if r.weather_condition == 'clear' else 0,
            'weather_rain': 1 if r.weather_condition == 'rain' else 0,
            'weather_fog': 1 if r.weather_condition == 'fog' else 0,
            'weather_snow': 1 if r.weather_condition == 'snow' else 0,
            'weather_storm': 1 if r.weather_condition == 'storm' else 0,
            'time_of_day': r.time_of_day,
            'network_congestion': r.network_congestion,
            'actual_travel_time': r.actual_travel_time,
            'delay': r.delay
        } for r in data])
        return df
        
    def train_model(self, historical_data):
        """Train the ETA prediction model"""
        df = self.prepare_features(historical_data)
        
        features = ['route_length', 'scheduled_speed', 'actual_speed', 
                   'weather_clear', 'weather_rain', 'weather_fog', 'weather_snow', 'weather_storm',
                   'time_of_day', 'network_congestion']
        
        X = df[features]
        y = df['actual_travel_time']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"ML Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        self.is_trained = True
        return test_score
        
    def predict_eta(self, route_length, scheduled_speed, current_conditions):
        """Predict ETA for a train given current conditions"""
        if not self.is_trained:
            return None
            
        features = np.array([[
            route_length,
            scheduled_speed,
            scheduled_speed,
            1 if current_conditions.get('weather', 'clear') == 'clear' else 0,
            1 if current_conditions.get('weather', 'clear') == 'rain' else 0,
            1 if current_conditions.get('weather', 'clear') == 'fog' else 0,
            1 if current_conditions.get('weather', 'clear') == 'snow' else 0,
            1 if current_conditions.get('weather', 'clear') == 'storm' else 0,
            current_conditions.get('time_of_day', 12),
            current_conditions.get('network_congestion', 0.5)
        ]])
        
        features_scaled = self.scaler.transform(features)
        predicted_time = self.model.predict(features_scaled)[0]
        
        return max(route_length * 3, int(predicted_time))

class ScheduleOptimizer:
    def __init__(self, traffic_system):
        self.traffic_system = traffic_system
        self.current_conditions = {
            'weather': 'clear',
            'time_of_day': 12,
            'network_congestion': 0.3
        }
        self.recommendations = {}  # Store recommendations by ID
        
    def update_conditions(self, conditions):
        self.current_conditions.update(conditions)
        
    def optimize_schedule(self, trains, ml_predictor):
        """Enhanced optimization with proper conflict resolution"""
        recommendations = []
        conflict_resolver = ConflictResolver(self.traffic_system)
        
        # Run conflict resolution first
        conflict_resolver.resolve_conflicts()
        
        for train_id, train in trains.items():
            if train['statusType'] in ['completed', 'cancelled']:
                continue
                
            route_length = len(train.get('route', []))
            
            # ML-based recommendations
            if ml_predictor.is_trained:
                predicted_eta = ml_predictor.predict_eta(
                    route_length, 
                    train['speed'], 
                    self.current_conditions
                )
                
                ideal_time = train.get('idealTravelTime', route_length * 5)
                predicted_delay = max(0, predicted_eta - ideal_time)
                
                if predicted_delay > 5:
                    rec_id = str(uuid.uuid4())
                    rec = OptimizationRecommendation(
                        id=rec_id,
                        type='speed_adjustment',
                        train_id=train_id,
                        train_number=train['number'],
                        priority=train.get('priority', 99),
                        reason=f'Predicted delay of {predicted_delay} ticks',
                        current_value=train['speed'],
                        recommended_value=min(160, int(train['speed'] * 1.2)),
                        confidence=0.85,
                        created_at=datetime.now()
                    )
                    recommendations.append(rec.__dict__)
                    self.recommendations[rec_id] = rec
                    
            # Conflict-based recommendations
            if train['waitingForBlock']:
                rec_id = str(uuid.uuid4())
                current_priority = train.get('priority', 99)
                
                # Check if train has been waiting too long
                if hasattr(train, 'waiting_since') and self.traffic_system.simulation_time - train.get('waiting_since', 0) > 10:
                    rec = OptimizationRecommendation(
                        id=rec_id,
                        type='priority_adjustment',
                        train_id=train_id,
                        train_number=train['number'],
                        priority=current_priority,
                        reason=f'Train waiting for {self.traffic_system.simulation_time - train.get("waiting_since", 0)} ticks',
                        current_value=current_priority,
                        recommended_value=max(1, current_priority - 10),
                        confidence=0.9,
                        created_at=datetime.now()
                    )
                    recommendations.append(rec.__dict__)
                    self.recommendations[rec_id] = rec
                    
        # Store recommendations for later retrieval
        return recommendations
        
    def apply_recommendation(self, recommendation_id: str, accept: bool) -> dict:
        """Apply or reject a specific recommendation"""
        if recommendation_id not in self.recommendations:
            return {"success": False, "message": "Recommendation not found"}
            
        recommendation = self.recommendations[recommendation_id]
        
        if accept:
            train = self.traffic_system.trains.get(recommendation.train_id)
            if not train:
                return {"success": False, "message": "Train not found"}
                
            if recommendation.type == 'speed_adjustment':
                old_speed = train['speed']
                train['speed'] = recommendation.recommended_value
                message = f"Speed adjusted for {train['number']}: {old_speed} → {train['speed']} km/h"
                
            elif recommendation.type == 'priority_adjustment':
                old_priority = train.get('priority', 99)
                train['priority'] = recommendation.recommended_value
                message = f"Priority adjusted for {train['number']}: P{old_priority} → P{train['priority']}"
                
            elif recommendation.type == 'route_change':
                # Implement route change logic
                message = f"Route changed for {train['number']}"
                
            else:
                return {"success": False, "message": "Unknown recommendation type"}
                
            # Mark as applied and log
            self.traffic_system.enhanced_metrics['recommendations_accepted'] += 1
            self.traffic_system.events.append(f"Optimization Applied: {message}")
            
            # Remove the recommendation
            del self.recommendations[recommendation_id]
            
            return {"success": True, "message": message}
        else:
            # Just remove the recommendation
            del self.recommendations[recommendation_id]
            return {"success": True, "message": "Recommendation rejected"}

class AuditLogger:
    def __init__(self):
        self.audit_log = []
        self.kpi_history = []
        
    def log_recommendation(self, recommendation, accepted=False):
        """Log optimization recommendations"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'recommendation',
            'data': recommendation,
            'accepted': accepted
        }
        self.audit_log.append(log_entry)
        
    def log_kpi(self, kpis):
        """Log KPI snapshots"""
        kpi_entry = {
            'timestamp': datetime.now().isoformat(),
            'kpis': kpis
        }
        self.kpi_history.append(kpi_entry)
        
    def get_recent_logs(self, limit=50):
        """Get recent audit logs"""
        return self.audit_log[-limit:]
        
    def get_kpi_history(self, hours=24):
        """Get KPI history for specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [entry for entry in self.kpi_history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff]

class EnhancedTrafficControlSystem:
    def __init__(self):
        # Original attributes
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
        
        # New ML and optimization components
        self.data_generator = SyntheticDataGenerator()
        self.ml_predictor = MLETAPredictor()
        self.optimizer = ScheduleOptimizer(self)
        self.audit_logger = AuditLogger()
        self.conflict_resolver = ConflictResolver(self)
        
        # Enhanced metrics
        self.enhanced_metrics = {
            'on_time_percentage': 100,
            'ml_accuracy': 0,
            'recommendations_accepted': 0,
            'total_recommendations': 0
        }
        
        # Initialize sections
        for section in TRACK_SECTIONS:
            sec_id = section['id']
            if section['type'] == 'block': 
                self.block_occupancy[sec_id] = None
            elif section['type'] == 'station' or section['type'] == 'junction': 
                self.station_platforms[sec_id] = {i: None for i in range(1, section['platforms'] + 1)}
                
        # Train ML model on startup
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize and train the ML model with synthetic data"""
        logger.info("Generating synthetic training data...")
        historical_data = self.data_generator.generate_historical_data(3000)
        
        logger.info("Training ML ETA prediction model...")
        accuracy = self.ml_predictor.train_model(historical_data)
        self.enhanced_metrics['ml_accuracy'] = accuracy
        
        logger.info(f"ML model initialized with accuracy: {accuracy:.3f}")

    async def add_websocket(self, websocket: WebSocket): 
        self.websocket_connections.add(websocket)
        
    async def remove_websocket(self, websocket: WebSocket): 
        self.websocket_connections.discard(websocket)

    async def broadcast_state(self):
        if not self.websocket_connections: 
            return
            
        state = self.get_system_state()
        state_json = json.dumps(state)
        disconnected_clients = set()
        
        for websocket in self.websocket_connections:
            try: 
                await websocket.send_text(state_json)
            except Exception: 
                disconnected_clients.add(websocket)
                
        for client in disconnected_clients: 
            self.websocket_connections.discard(client)

    def inject_delay(self, train_id: str, delay_minutes: int):
        """Inject artificial delay into a train"""
        if train_id in self.trains:
            train = self.trains[train_id]
            train['speed'] = max(10, train['speed'] - (delay_minutes * 2))
            train['injected_delay'] = delay_minutes
            
            self.events.append(f"Delay Injected: {train['number']} delayed by {delay_minutes}min")
            return True
        return False

    def apply_optimization_recommendation(self, recommendation_id: str, accept: bool):
        """Apply an optimization recommendation using the optimizer"""
        return self.optimizer.apply_recommendation(recommendation_id, accept)

    def get_ml_predictions(self):
        """Get ML ETA predictions for all active trains"""
        predictions = {}
        
        for train_id, train in self.trains.items():
            if train['statusType'] in ['running', 'scheduled']:
                route_length = len(train.get('route', []))
                
                predicted_eta = self.ml_predictor.predict_eta(
                    route_length,
                    train['speed'],
                    self.optimizer.current_conditions
                )
                
                if predicted_eta:
                    ideal_time = train.get('idealTravelTime', route_length * 5)
                    predictions[train_id] = {
                        'predicted_eta': predicted_eta,
                        'ideal_time': ideal_time,
                        'predicted_delay': max(0, predicted_eta - ideal_time),
                        'confidence': 0.85 + random.uniform(-0.1, 0.1)  # Add some variance
                    }
                    
        return predictions

    def get_optimization_recommendations(self):
        """Get current optimization recommendations"""
        recommendations = self.optimizer.optimize_schedule(self.trains, self.ml_predictor)
        
        # Update total recommendations count
        self.enhanced_metrics['total_recommendations'] = len(self.optimizer.recommendations)
        
        return recommendations

    def _update_enhanced_metrics(self):
        """Update enhanced KPIs including ML accuracy and on-time performance"""
        # Calculate on-time percentage
        if self.completed_train_stats:
            on_time_trains = sum(1 for s in self.completed_train_stats 
                               if s['actual_time'] - s['ideal_time'] <= 3)
            self.enhanced_metrics['on_time_percentage'] = (on_time_trains / len(self.completed_train_stats)) * 100
        else:
            self.enhanced_metrics['on_time_percentage'] = 100
            
        # Log KPIs
        combined_metrics = {**self.metrics, **self.enhanced_metrics}
        self.audit_logger.log_kpi(combined_metrics)

    def find_all_paths(self, start: str, end: str, max_paths: int = 5) -> List[List[str]]:
        """Find multiple paths between two nodes for rerouting"""
        all_paths = []
        
        def dfs(current_path, visited, target):
            if len(all_paths) >= max_paths:
                return
                
            current_node = current_path[-1]
            if current_node == target:
                all_paths.append(current_path.copy())
                return
                
            if len(current_path) > 10:  # Prevent infinite loops
                return
                
            for neighbor in GRAPH.get(current_node, {}):
                if neighbor not in visited or neighbor == target:
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    current_path.append(neighbor)
                    dfs(current_path, new_visited, target)
                    current_path.pop()
        
        visited = {start}
        dfs([start], visited, end)
        
        # Sort by path length
        all_paths.sort(key=len)
        return all_paths

    def occupy_section(self, section_id: str, train_id: str):
        if section_id in self.block_occupancy: 
            self.block_occupancy[section_id] = train_id
        elif section_id in self.station_platforms:
            for p_num, occupant in self.station_platforms[section_id].items():
                if occupant is None: 
                    self.station_platforms[section_id][p_num] = train_id
                    break
                
    def release_section(self, section_id: str, train_id: str):
        if section_id in self.block_occupancy and self.block_occupancy[section_id] == train_id: 
            self.block_occupancy[section_id] = None
        elif section_id in self.station_platforms:
            for p_num, occupant in self.station_platforms[section_id].items():
                if occupant == train_id: 
                    self.station_platforms[section_id][p_num] = None
                    break

    def calculate_travel_time(self, train_speed: int, is_station_pass_through=False) -> int:
        if is_station_pass_through: 
            return 1
        speed_factor = max(0.5, min(2.0, 100 / train_speed))
        return max(3, int(5 * speed_factor))

    def calculate_ideal_travel_time(self, route: List[str], speed: int, stops: List[str]) -> int:
        ideal_time = 0
        for section_id in route:
            is_station = any(s['id'] == section_id and s['type'] in ['station', 'junction'] for s in TRACK_SECTIONS)
            is_stop = section_id in stops
            ideal_time += self.calculate_travel_time(speed, is_station and not is_stop)
            if is_stop: 
                ideal_time += 5
        return ideal_time
    
    def add_train(self, train_data: dict):
        train_id, start_stn, dest_stn = train_data['id'], f"STN_{train_data['start']}", f"STN_{train_data['destination']}"
        route = self.find_shortest_path(start_stn, dest_stn)
        if not route: 
            return None
        stops = train_data.get('stops', [])
        ideal_time = self.calculate_ideal_travel_time(route, train_data['speed'], stops)
        train = {
            'id': train_id, 'name': train_data['name'], 'number': train_data['number'], 'section': route[0],
            'speed': train_data['speed'], 'destination': train_data['destination'], 'status': 'Scheduled',
            'statusType': 'scheduled', 'route': route, 'departureTime': train_data.get('departureTime', 0),
            'waitingForBlock': False, 'stops': stops, 'atStation': False, 'dwellTimeStart': 0, 
            'idealTravelTime': ideal_time, 'priority': train_data.get('priority', 99),
            'injected_delay': 0, 'waiting_since': None
        }
        self.trains[train_id] = train
        self.train_progress[train_id] = {'currentRouteIndex': 0, 'lastMoveTime': train_data.get('departureTime', 0)}
        self.occupy_section(route[0], train_id)
        return train

    def _update_metrics(self):
        completed_count = len(self.completed_train_stats)
        self.metrics["throughput"] = (completed_count / self.simulation_time) * 60 if self.simulation_time > 0 else 0
        self.metrics["avgDelay"] = sum(s['actual_time'] - s['ideal_time'] for s in self.completed_train_stats) / completed_count if completed_count > 0 else 0
        occupied_sections = sum(1 for o in self.block_occupancy.values() if o is not None)
        total_platforms = 0
        for p in self.station_platforms.values():
            occupied_sections += sum(1 for o in p.values() if o is not None)
            total_platforms += len(p)
        total_sections = len(self.block_occupancy) + total_platforms
        self.metrics["utilization"] = (occupied_sections / total_sections) * 100 if total_sections > 0 else 0
        running_trains = [t for t in self.trains.values() if t['statusType'] == 'running' and not t['waitingForBlock']]
        self.metrics["avgSpeed"] = sum(t['speed'] for t in running_trains) / len(running_trains) if running_trains else 0
        
        # Update enhanced metrics
        self._update_enhanced_metrics()

    async def update_simulation(self):
        self.events.clear()
        if self.is_running:
            self.simulation_time += 1
            
            # Run conflict resolution every few ticks
            if self.simulation_time % 3 == 0:
                self.conflict_resolver.resolve_conflicts()
            
            await asyncio.gather(*(self._update_train(tid) for tid in list(self.trains.keys())))
            self._update_metrics()
            await self.broadcast_state()

    async def _update_train(self, train_id: str):
        train = self.trains[train_id]
        progress = self.train_progress[train_id]
        was_waiting = train['waitingForBlock']
        
        # Don't update if train hasn't departed yet
        if self.simulation_time < train['departureTime']: 
            return
        
        current_idx = progress['currentRouteIndex']
        route = train['route']
        
        # Check if train has completed its journey
        if current_idx >= len(route) - 1 and train['statusType'] != 'completed':
            final_section = route[-1]
            self.release_section(final_section, train_id)
            
            train.update({'status': 'Arrived', 'statusType': 'completed'})
            self.completed_train_stats.append({
                'id': train_id, 
                'ideal_time': train['idealTravelTime'], 
                'actual_time': self.simulation_time - train['departureTime']
            })
            self.events.append(f"Arrival: {train['number']} at {final_section}.")
            return
            
        if train['statusType'] == 'completed': 
            return
        
        train['statusType'] = 'running'
        current_section, next_section = train['route'][current_idx], train['route'][current_idx + 1]
        
        # Handle station stops
        is_at_station = any(s['id'] == current_section and s['type'] in ['station', 'junction'] for s in TRACK_SECTIONS)
        is_stop = current_section in train.get('stops', [])
        
        if is_at_station and is_stop:
            if not train['atStation']: 
                train['atStation'], train['dwellTimeStart'] = True, self.simulation_time
                self.events.append(f"Halt: {train['number']} at {current_section}.")
            if self.simulation_time - train['dwellTimeStart'] < 5: 
                train['status'] = f"Halting at {current_section}"
                return
        
        if not is_at_station and train['atStation']: 
            train['atStation'] = False
        
        required_time = self.calculate_travel_time(train['speed'], is_at_station and not is_stop)
        
        if self.simulation_time - progress['lastMoveTime'] >= required_time:
            if self.is_section_available(next_section, train_id):
                self.release_section(current_section, train_id)
                self.occupy_section(next_section, train_id)
                train.update({'section': next_section, 'status': f"En route", 'waitingForBlock': False, 'waiting_since': None})
                progress.update({'currentRouteIndex': current_idx + 1, 'lastMoveTime': self.simulation_time})
                
                if was_waiting:
                    self.events.append(f"Movement: {train['number']} resumed to {next_section}")
            else:
                if not was_waiting:
                    train['waiting_since'] = self.simulation_time
                    
                train.update({'waitingForBlock': True, 'status': f"Waiting for {next_section}"})
                
                if not was_waiting:
                    occupying_train_id = self.block_occupancy.get(next_section)
                    if not occupying_train_id:
                        for platform, occupant in self.station_platforms.get(next_section, {}).items():
                            if occupant:
                                occupying_train_id = occupant
                                break
                    
                    if occupying_train_id:
                        occupying_train = self.trains.get(occupying_train_id)
                        if occupying_train:
                            event_message = f"Conflict: {train['number']} waits for {occupying_train['number']}."
                            if event_message not in self.events: 
                                self.events.append(event_message)

    def get_system_state(self): 
        state = {
            "trains": list(self.trains.values()), 
            "blockOccupancy": self.block_occupancy, 
            "stationPlatforms": self.station_platforms, 
            "simulationTime": self.simulation_time, 
            "isRunning": self.is_running, 
            "trainProgress": self.train_progress, 
            "metrics": self.metrics, 
            "events": self.events,
            "enhancedMetrics": self.enhanced_metrics,
            "mlPredictions": self.get_ml_predictions(),
            "optimizationRecommendations": self.get_optimization_recommendations()
        }
        return state

    def find_shortest_path(self, start_node: str, end_node: str) -> List[str]:
        distances={node: float('inf') for node in GRAPH}
        distances[start_node]=0
        pq=[(0, start_node)]
        prev_nodes={node: None for node in GRAPH}
        while pq:
            dist, current=heapq.heappop(pq)
            if dist > distances[current]: 
                continue
            if current == end_node: 
                break
            for neighbor, weight in GRAPH[current].items():
                new_dist=dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor]=new_dist
                    prev_nodes[neighbor]=current
                    heapq.heappush(pq, (new_dist, neighbor))
        path=[]
        current=end_node
        while current is not None: 
            path.insert(0, current)
            current=prev_nodes[current]
        return path if path and path[0] == start_node else []

    def is_section_available(self, section_id: str, train_id: str) -> bool:
        if section_id in self.block_occupancy: 
            return self.block_occupancy[section_id] is None
        elif section_id in self.station_platforms: 
            return any(p is None for p in self.station_platforms[section_id].values())
        return False
        
    def reset_simulation(self):
        self.is_running, self.simulation_time = False, 0
        self.trains.clear()
        self.train_progress.clear()
        self.completed_train_stats.clear()
        self.events.clear()
        self.enhanced_metrics = {
            'on_time_percentage': 100,
            'ml_accuracy': self.enhanced_metrics.get('ml_accuracy', 0),
            'recommendations_accepted': 0,
            'total_recommendations': 0
        }
        
        for sec_id in self.block_occupancy: 
            self.block_occupancy[sec_id] = None
        for stn_id in self.station_platforms:
            for p_num in self.station_platforms[stn_id]: 
                self.station_platforms[stn_id][p_num] = None
                
        # Clear optimizer recommendations
        self.optimizer.recommendations.clear()
        
        add_default_trains(self)
        logger.info("Enhanced simulation reset with conflict resolution")
        
    def start_simulation(self): 
        self.is_running = True
        
    def pause_simulation(self): 
        self.is_running = False

# Initialize the enhanced system
traffic_system = EnhancedTrafficControlSystem()

# Pydantic models
class TrainData(BaseModel): 
    id: str
    name: str
    number: str
    start: str
    destination: str
    speed: int
    departureTime: int = 0
    stops: List[str] = []
    priority: int = 99

class SimulationControl(BaseModel): 
    action: str

class DelayInjection(BaseModel):
    train_id: str
    delay_minutes: int

class OptimizationRequest(BaseModel):
    recommendation_id: str
    accept: bool

class ConditionUpdate(BaseModel):
    weather: Optional[str] = None
    time_of_day: Optional[int] = None
    network_congestion: Optional[float] = None

def add_default_trains(system: EnhancedTrafficControlSystem):
    """Add expanded set of default trains to showcase the larger network"""
    default_trains = [
        # Express services with different priorities
        {'id': 'T1', 'number': '12301', 'name': 'Metro Express', 'start': 'A', 'destination': 'D', 
         'speed': 120, 'departureTime': 0, 'stops': ['STN_A', 'STN_C', 'STN_D'], 'priority': 5},
        {'id': 'T2', 'number': '12302', 'name': 'Northern Express', 'start': 'E', 'destination': 'G', 
         'speed': 110, 'departureTime': 3, 'stops': ['STN_E', 'STN_F', 'STN_G'], 'priority': 8},
        {'id': 'T3', 'number': '12303', 'name': 'Summit Special', 'start': 'H', 'destination': 'I', 
         'speed': 100, 'departureTime': 5, 'stops': ['STN_H', 'STN_I'], 'priority': 12},
        
        # Local services
        {'id': 'T4', 'number': '22401', 'name': 'Local Service', 'start': 'A', 'destination': 'B', 
         'speed': 80, 'departureTime': 8, 'stops': ['STN_A', 'STN_B'], 'priority': 25},
        {'id': 'T5', 'number': '22402', 'name': 'Bay Local', 'start': 'J', 'destination': 'L', 
         'speed': 70, 'departureTime': 10, 'stops': ['STN_J', 'STN_K', 'STN_L'], 'priority': 30},
        
        # Freight services - lower priority
        {'id': 'T6', 'number': '32601', 'name': 'Freight Heavy', 'start': 'A', 'destination': 'L', 
         'speed': 50, 'departureTime': 12, 'stops': ['STN_A', 'STN_L'], 'priority': 60},
        
        # Cross-network services
        {'id': 'T7', 'number': '42801', 'name': 'Cross Network', 'start': 'H', 'destination': 'L', 
         'speed': 90, 'departureTime': 15, 'stops': ['STN_H', 'STN_F', 'STN_B', 'STN_K', 'STN_L'], 'priority': 20},
    ]
    
    for train_data in default_trains: 
        system.add_train(train_data)

@app.on_event("startup")
async def startup_event(): 
    add_default_trains(traffic_system)
    asyncio.create_task(simulation_loop())

async def simulation_loop():
    while True:
        if traffic_system.is_running: 
            await traffic_system.update_simulation()
        await asyncio.sleep(1.2)  # Slightly faster simulation

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
    action = control.action
    if action == "start": 
        traffic_system.start_simulation()
    elif action == "pause": 
        traffic_system.pause_simulation()
    elif action == "reset": 
        traffic_system.reset_simulation()
    else: 
        raise HTTPException(status_code=400, detail="Invalid action")
    await traffic_system.broadcast_state()
    return {"status": "success", "action": action}

@app.post("/add-train")
async def add_train_endpoint(train: TrainData):
    new_train = traffic_system.add_train(train.dict())
    if new_train is None: 
        raise HTTPException(status_code=400, detail="Could not create train")
    await traffic_system.broadcast_state()
    return {"status": "success", "train": new_train}

# Enhanced ML and Optimization endpoints
@app.post("/inject-delay")
async def inject_delay(delay_data: DelayInjection):
    """Inject artificial delay into a train for testing"""
    success = traffic_system.inject_delay(delay_data.train_id, delay_data.delay_minutes)
    if not success:
        raise HTTPException(status_code=404, detail="Train not found")
    
    await traffic_system.broadcast_state()
    return {"status": "success", "message": f"Delay of {delay_data.delay_minutes} minutes injected"}

@app.post("/apply-optimization")
async def apply_optimization(opt_request: OptimizationRequest):
    """Apply or reject an optimization recommendation"""
    result = traffic_system.apply_optimization_recommendation(opt_request.recommendation_id, opt_request.accept)
    
    if result["success"]:
        await traffic_system.broadcast_state()
    else:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return {"status": "success", "message": result["message"]}

@app.post("/update-conditions")
async def update_conditions(conditions: ConditionUpdate):
    """Update current operational conditions"""
    condition_dict = conditions.dict(exclude_unset=True)
    traffic_system.optimizer.update_conditions(condition_dict)
    
    return {"status": "success", "message": "Conditions updated", "conditions": condition_dict}

@app.get("/ml-predictions")
async def get_ml_predictions():
    """Get current ML ETA predictions for all trains"""
    predictions = traffic_system.get_ml_predictions()
    return {"predictions": predictions}

@app.get("/optimization-recommendations")
async def get_optimization_recommendations():
    """Get current optimization recommendations"""
    recommendations = traffic_system.get_optimization_recommendations()
    return {"recommendations": recommendations}

@app.get("/audit-logs")
async def get_audit_logs(limit: int = 50):
    """Get recent audit logs"""
    logs = traffic_system.audit_logger.get_recent_logs(limit)
    return {"logs": logs}

@app.get("/enhanced-metrics")
async def get_enhanced_metrics():
    """Get enhanced metrics including ML performance"""
    return {
        "basic_metrics": traffic_system.metrics,
        "enhanced_metrics": traffic_system.enhanced_metrics,
        "ml_accuracy": traffic_system.enhanced_metrics.get('ml_accuracy', 0),
        "recommendations_ratio": (
            traffic_system.enhanced_metrics.get('recommendations_accepted', 0) / 
            max(1, traffic_system.enhanced_metrics.get('total_recommendations', 1))
        )
    }

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)