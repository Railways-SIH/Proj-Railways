# backend/core/system.py

from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, HTTPException
import heapq
import asyncio
import json
import logging
import math

# FROM: from backend.core.const import TRACK_SECTIONS, GRAPH, _convert_ticks_to_minutes
from .const import TRACK_SECTIONS, GRAPH, _convert_ticks_to_minutes

# FROM: from backend.ml.predictor import SyntheticDataGenerator, MLETAPredictor
from ..ml.predictor import SyntheticDataGenerator, MLETAPredictor # .. for one level up

# FROM: from backend.ml.optimizer import ScheduleOptimizer
from ..ml.optimizer import ScheduleOptimizer # .. for one level up

# FROM: from backend.logging.audit import AuditLogger
from ..logging.audit import AuditLogger # .. for one level up

logger = logging.getLogger(__name__)

class EnhancedTrafficControlSystem:
    def __init__(self):
        # Original attributes
        self.trains: Dict[str, Dict[str, Any]] = {}
        self.block_occupancy: Dict[str, Optional[str]] = {}
        self.station_platforms: Dict[str, Dict[int, Optional[str]]] = {}
        self.simulation_time = 0 # In Ticks
        self.is_running = False
        self.train_progress: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Set[WebSocket] = set()
        self.metrics = { "throughput": 0, "avgDelay": 0, "utilization": 0, "avgSpeed": 0 }
        self.completed_train_stats: List[Dict[str, Any]] = []
        self.events: List[str] = []
        
        # New ML and optimization components
        self.data_generator = SyntheticDataGenerator()
        self.ml_predictor = MLETAPredictor()
        self.optimizer = ScheduleOptimizer()
        self.audit_logger = AuditLogger()
        
        # Enhanced metrics
        self.enhanced_metrics = {
            'on_time_percentage': 0,
            'ml_accuracy': 0,
            'recommendations_accepted': 0,
            'total_recommendations': 0
        }
        
        # Initialize sections
        for section in TRACK_SECTIONS:
            sec_id = section['id']
            if section['type'] == 'block': 
                self.block_occupancy[sec_id] = None
            elif section['type'] == 'station': 
                self.station_platforms[sec_id] = {i: None for i in range(1, section['platforms'] + 1)}
                
        # Train ML model on startup
        self._initialize_ml_model()
    # In core/system.py, inside the EnhancedTrafficControlSystem class

    def reject_optimization_recommendation(self, recommendation_id: str):
        """Logs a rejected recommendation."""
        # This is a placeholder for more complex logic if needed in the future.
        # For now, it just ensures the audit log is aware of the rejection.
        self.audit_logger.log_recommendation({"id": recommendation_id}, accepted=False)
        return True
        
    def _initialize_ml_model(self):
        """Initialize and train the ML model with synthetic data"""
        logger.info("Generating synthetic training data...")
        historical_data = self.data_generator.generate_historical_data(3000)
        
        logger.info("Training ML ETA prediction model...")
        accuracy = self.ml_predictor.train_model(historical_data)
        self.enhanced_metrics['ml_accuracy'] = round(accuracy, 4)
        
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

    def inject_delay(self, train_id: str, delay_minutes: int) -> bool:
        """Inject artificial delay into a train (speed reduction is proportional to delay)"""
        if train_id in self.trains:
            train = self.trains[train_id]
            speed_reduction = delay_minutes * 3
            train['speed'] = max(10, train['speed'] - speed_reduction)
            train['injected_delay'] = delay_minutes
            self.events.append(f"Delay Injected: {train['number']} slowed, {delay_minutes} min delay simulated.")
            return True
        return False

    def apply_optimization_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Apply an optimization recommendation"""
        train_id = recommendation['train_id']
        if train_id not in self.trains:
            return False
            
        train = self.trains[train_id]
        
        if recommendation['type'] == 'speed_adjustment':
            old_speed = train['speed']
            train['speed'] = recommendation['recommended_speed']
            self.events.append(f"Speed Adjusted: {train['number']} {old_speed}→{train['speed']} km/h")
            
        elif recommendation['type'] == 'priority_adjustment':
            old_priority = train.get('priority', 99)
            train['priority'] = recommendation['recommended_priority']
            self.events.append(f"Priority Adjusted: {train['number']} P{old_priority}→P{train['priority']}")
            
        self.audit_logger.log_recommendation(recommendation, accepted=True)
        self.enhanced_metrics['recommendations_accepted'] += 1
        return True

    def _calculate_confidence(self, predicted_delay_ticks: int) -> float:
        """Dynamically calculate confidence based on delay magnitude (in ticks)"""
        max_delay_for_confidence = 50
        
        if predicted_delay_ticks <= 0:
            return 0.95
        
        decay = (min(predicted_delay_ticks, max_delay_for_confidence) / max_delay_for_confidence) * 0.45
        confidence = max(0.5, 0.95 - decay)
        
        return round(confidence, 2)

    def get_ml_predictions(self) -> Dict[str, Dict[str, float]]:
        """Get ML ETA predictions (in minutes) for all active trains"""
        predictions = {}
        
        for train_id, train in self.trains.items():
            if train['statusType'] in ['running', 'scheduled']:
                current_idx = self.train_progress.get(train_id, {}).get('currentRouteIndex', 0)
                remaining_route_length = len(train.get('route', [])) - current_idx
                
                if remaining_route_length <= 0:
                    continue
                
                predicted_eta_ticks = self.ml_predictor.predict_eta(
                    remaining_route_length,
                    train['speed'],
                    self.optimizer.current_conditions
                )
                
                if predicted_eta_ticks:
                    remaining_route = train['route'][current_idx:]
                    remaining_stops = [s for s in train.get('stops', []) if s in remaining_route]
                    
                    ideal_time_ticks = self.calculate_ideal_travel_time(
                        remaining_route, train['speed'], remaining_stops
                    )
                    
                    predicted_delay_ticks = predicted_eta_ticks - ideal_time_ticks
                    
                    predictions[train_id] = {
                        'predicted_eta': _convert_ticks_to_minutes(predicted_eta_ticks),
                        'ideal_time': _convert_ticks_to_minutes(ideal_time_ticks),
                        'predicted_delay': _convert_ticks_to_minutes(predicted_delay_ticks),
                        'confidence': self._calculate_confidence(predicted_delay_ticks)
                    }
                    
        return predictions

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations"""
        recommendations = self.optimizer.optimize_schedule(self.trains, self.ml_predictor)
        
        for rec in recommendations:
            self.audit_logger.log_recommendation(rec, accepted=False)
            self.enhanced_metrics['total_recommendations'] += 1
            
        return recommendations

    def _update_enhanced_metrics(self):
        """Update enhanced KPIs including ML accuracy and on-time performance"""
        if self.completed_train_stats:
            on_time_trains = sum(1 for s in self.completed_train_stats 
                                 if s['delay_ticks'] <= 3)
            self.enhanced_metrics['on_time_percentage'] = (on_time_trains / len(self.completed_train_stats)) * 100
        else:
            self.enhanced_metrics['on_time_percentage'] = 100
            
        combined_metrics = {**self.metrics, **self.enhanced_metrics}
        self.audit_logger.log_kpi(combined_metrics)

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
            is_station, is_stop = section_id.startswith('STN_'), section_id in stops
            ideal_time += self.calculate_travel_time(speed, is_station_pass_through=(is_station and not is_stop))
            if is_stop: 
                ideal_time += 5 
        return ideal_time
    
    def add_train(self, train_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            'injected_delay': 0 
        }
        self.trains[train_id] = train
        self.train_progress[train_id] = {'currentRouteIndex': 0, 'lastMoveTime': train_data.get('departureTime', 0)}
        self.occupy_section(route[0], train_id)
        return train

    def _update_metrics(self):
        completed_count = len(self.completed_train_stats)
        simulation_time_min = _convert_ticks_to_minutes(self.simulation_time)
        
        self.metrics["throughput"] = (completed_count / simulation_time_min) * 60 if simulation_time_min > 0 else 0
        
        total_delay_ticks = sum(s['delay_ticks'] for s in self.completed_train_stats)
        self.metrics["avgDelay"] = _convert_ticks_to_minutes(total_delay_ticks) / completed_count if completed_count > 0 else 0
        
        occupied_sections = sum(1 for o in self.block_occupancy.values() if o is not None)
        total_platforms = 0
        for p in self.station_platforms.values():
            occupied_sections += sum(1 for o in p.values() if o is not None)
            total_platforms += len(p)
        
        total_sections = len(self.block_occupancy) + total_platforms
        self.metrics["utilization"] = (occupied_sections / total_sections) * 100 if total_sections > 0 else 0
        
        running_trains = [t for t in self.trains.values() if t['statusType'] == 'running' and not t['waitingForBlock']]
        self.metrics["avgSpeed"] = sum(t['speed'] for t in running_trains) / len(running_trains) if running_trains else 0
        
        self._update_enhanced_metrics()

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
        
        if self.simulation_time < train['departureTime']: 
            return
        
        current_idx = progress['currentRouteIndex']
        route = train['route']
        
        # Completion check
        if current_idx >= len(route) - 1 and train['statusType'] != 'completed':
            final_section = route[-1]
            self.release_section(final_section, train_id)
            actual_time_ticks = self.simulation_time - train['departureTime']
            ideal_time_ticks = train['idealTravelTime']
            delay_ticks = actual_time_ticks - ideal_time_ticks

            train.update({'status': 'Arrived', 'statusType': 'completed'})
            self.completed_train_stats.append({
                'id': train_id, 'ideal_time_ticks': ideal_time_ticks, 'actual_time_ticks': actual_time_ticks, 
                'delay_ticks': delay_ticks, 'ideal_time_min': _convert_ticks_to_minutes(ideal_time_ticks),
                'actual_time_min': _convert_ticks_to_minutes(actual_time_ticks), 'delay_min': _convert_ticks_to_minutes(delay_ticks) 
            })
            delay_min_display = _convert_ticks_to_minutes(delay_ticks)
            self.events.append(f"Arrival: {train['number']} at {final_section}. Delay: {delay_min_display:.2f} min.")
            return
            
        if train['statusType'] == 'completed': 
            return
        
        train['statusType'] = 'running'
        current_section = train['route'][current_idx]
        next_section = train['route'][current_idx + 1]
        is_at_station, is_stop = current_section.startswith('STN_'), current_section in train.get('stops', [])
        
        # Station Halting Logic
        if is_at_station and is_stop:
            if not train['atStation']: 
                train['atStation'], train['dwellTimeStart'] = True, self.simulation_time
                self.events.append(f"Halt: {train['number']} at {current_section}.")
            if self.simulation_time - train['dwellTimeStart'] < 5: 
                train['status'] = f"Halting at {current_section}"
                return
        if not current_section.startswith('STN_') and train['atStation']: 
            train['atStation'] = False
        
        required_time = self.calculate_travel_time(train['speed'], is_at_station and not is_stop)
        
        if self.simulation_time - progress['lastMoveTime'] >= required_time:
            if self.is_section_available(next_section, train_id):
                # Move train to next section
                self.release_section(current_section, train_id)
                self.occupy_section(next_section, train_id)
                train.update({'section': next_section, 'status': "En route", 'waitingForBlock': False})
                progress.update({'currentRouteIndex': current_idx + 1, 'lastMoveTime': self.simulation_time})
            else:
                # Wait for next section
                train.update({'waitingForBlock': True, 'status': f"Waiting for {next_section}"})
                if not was_waiting:
                    occupying_train_id = self.block_occupancy.get(next_section) or next((occ for occ in self.station_platforms.get(next_section, {}).values() if occ), None)
                    occupying_train = self.trains.get(occupying_train_id)
                    if occupying_train:
                        event_message = f"Conflict: {train['number']} waits for {occupying_train['number']}."
                        if event_message not in self.events: 
                            self.events.append(event_message)

    def get_system_state(self) -> Dict[str, Any]: 

        full_audit_history = self.audit_logger.get_recent_logs(limit=200)
        state = {
            "trains": list(self.trains.values()), 
            "blockOccupancy": self.block_occupancy, 
            "stationPlatforms": self.station_platforms, 
            "simulationTime": self.simulation_time, 
            "simulationTimeMinutes": _convert_ticks_to_minutes(self.simulation_time),
            "isRunning": self.is_running, 
            "trainProgress": self.train_progress, 
            "metrics": self.metrics, 
            "events": self.events,
            "auditLogHistory": full_audit_history,
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
        self.enhanced_metrics['recommendations_accepted'] = 0
        self.enhanced_metrics['total_recommendations'] = 0
        for sec_id in self.block_occupancy: 
            self.block_occupancy[sec_id] = None
        for stn_id in self.station_platforms:
            for p_num in self.station_platforms[stn_id]: 
                self.station_platforms[stn_id][p_num] = None
        add_default_trains(self)
        logger.info("Enhanced simulation reset")
        
    def start_simulation(self): 
        self.is_running = True
        
    def pause_simulation(self): 
        self.is_running = False

# Helper function for initial setup
def add_default_trains(system: EnhancedTrafficControlSystem):
    """Add expanded set of default trains to showcase the larger network"""
    default_trains = [
        # Express services
        {'id': 'T1', 'number': '12301', 'name': 'Metro Express', 'start': 'A', 'destination': 'D', 
         'speed': 140, 'departureTime': 0, 'stops': ['STN_A', 'STN_C', 'STN_D'], 'priority': 5},
        {'id': 'T2', 'number': '12302', 'name': 'Northern Express', 'start': 'E', 'destination': 'G', 
         'speed': 130, 'departureTime': 2, 'stops': ['STN_E', 'STN_F', 'STN_G'], 'priority': 10},
        {'id': 'T3', 'number': '12303', 'name': 'Summit Special', 'start': 'H', 'destination': 'I', 
         'speed': 120, 'departureTime': 4, 'stops': ['STN_H', 'STN_I'], 'priority': 15},
        
        # Local services
        {'id': 'T4', 'number': '22401', 'name': 'Local Service', 'start': 'A', 'destination': 'B', 
         'speed': 80, 'departureTime': 6, 'stops': ['STN_A', 'STN_B'], 'priority': 25},
        {'id': 'T5', 'number': '22402', 'name': 'Bay Local', 'start': 'J', 'destination': 'L', 
         'speed': 70, 'departureTime': 8, 'stops': ['STN_J', 'STN_K', 'STN_L'], 'priority': 30},
        
        # Freight services
        {'id': 'T6', 'number': '32601', 'name': 'Freight Heavy', 'start': 'A', 'destination': 'L', 
         'speed': 50, 'departureTime': 10, 'stops': ['STN_A', 'STN_L'], 'priority': 50},
        {'id': 'T7', 'number': '32602', 'name': 'Cargo Express', 'start': 'E', 'destination': 'D', 
         'speed': 60, 'departureTime': 12, 'stops': ['STN_E', 'STN_D'], 'priority': 40},
        
        # Cross-network services
        {'id': 'T8', 'number': '42801', 'name': 'Cross Network', 'start': 'H', 'destination': 'J', 
         'speed': 100, 'departureTime': 14, 'stops': ['STN_H', 'STN_F', 'STN_B', 'STN_K', 'STN_L'], 'priority': 20},
        {'id': 'T9', 'number': '42802', 'name': 'Circle Line', 'start': 'A', 'destination': 'A', 
         'speed': 90, 'departureTime': 16, 'stops': ['STN_A', 'STN_E', 'STN_G', 'STN_C', 'STN_A'], 'priority': 35},
        
        # Peak hour services
        {'id': 'T10', 'number': '52901', 'name': 'Peak Express', 'start': 'G', 'destination': 'A', 
         'speed': 110, 'departureTime': 18, 'stops': ['STN_G', 'STN_C', 'STN_A'], 'priority': 8},
        {'id': 'T11', 'number': '52902', 'name': 'Commuter Rush', 'start': 'I', 'destination': 'J', 
         'speed': 95, 'departureTime': 20, 'stops': ['STN_I', 'STN_H', 'STN_F', 'STN_B', 'STN_J'], 'priority': 12},
        
        # Additional services for network testing
        {'id': 'T12', 'number': '62001', 'name': 'Network Test A', 'start': 'D', 'destination': 'D', 
         'speed': 85, 'departureTime': 22, 'stops': ['STN_D', 'STN_C', 'STN_G', 'STN_F', 'STN_D'], 'priority': 45},
    ]
    
    for train_data in default_trains: 
        system.add_train(train_data)