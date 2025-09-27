import heapq
from typing import Dict, List, Set, Tuple
from datetime import datetime
import asyncio
import json
import logging
from fastapi import WebSocket
from optimizers import SyntheticDataGenerator, MLETAPredictor, ScheduleOptimizer, AuditLogger

# Initialize logger
logger = logging.getLogger(__name__)

# Expanded track sections with more stations and blocks
TRACK_SECTIONS = [
    # Main line stations and blocks
    {'id': 'STN_A', 'type': 'station', 'name': 'Central Station A', 'station': 'A', 'platforms': 4}, 
    {'id': 'BLOCK_A1', 'type': 'block', 'name': 'Block A1'},
    {'id': 'BLOCK_A2', 'type': 'block', 'name': 'Block A2'}, 
    {'id': 'STN_B', 'type': 'station', 'name': 'Junction B', 'station': 'B', 'platforms': 3},
    {'id': 'BLOCK_B1', 'type': 'block', 'name': 'Block B1'}, 
    {'id': 'BLOCK_B2', 'type': 'block', 'name': 'Block B2'},
    {'id': 'STN_C', 'type': 'station', 'name': 'Metro C', 'station': 'C', 'platforms': 3}, 
    {'id': 'BLOCK_C1', 'type': 'block', 'name': 'Block C1'},
    {'id': 'BLOCK_C2', 'type': 'block', 'name': 'Block C2'}, 
    {'id': 'STN_D', 'type': 'station', 'name': 'Terminal D', 'station': 'D', 'platforms': 2},
    
    # Northern branch
    {'id': 'STN_E', 'type': 'station', 'name': 'North Hub E', 'station': 'E', 'platforms': 3}, 
    {'id': 'BLOCK_E1', 'type': 'block', 'name': 'Block E1'},
    {'id': 'BLOCK_E2', 'type': 'block', 'name': 'Block E2'}, 
    {'id': 'STN_F', 'type': 'station', 'name': 'Express F', 'station': 'F', 'platforms': 2},
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
    {'id': 'STN_K', 'type': 'station', 'name': 'Coast K', 'station': 'K', 'platforms': 2},
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

# Expanded graph with all new connections
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
        self.optimizer = ScheduleOptimizer()
        self.audit_logger = AuditLogger()
        
        # Enhanced metrics
        self.enhanced_metrics = {
            'on_time_percentage': 0,
            'ml_accuracy': 0,
            'recommendations_accepted': 0,
            'total_recommendations': 0
        }
        
        # Disruption tracking
        self.disrupted_sections: Set[str] = set()
        self.disrupted_trains: Set[str] = set()
        
        # Initialize sections
        for section in TRACK_SECTIONS:
            sec_id = section['id']
            if section['type'] == 'block': 
                self.block_occupancy[sec_id] = None
            elif section['type'] == 'station': 
                self.station_platforms[sec_id] = {i: None for i in range(1, section['platforms'] + 1)}
                
        # Train ML model on startup
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize and train the ML model with synthetic data"""
        try:
            logger.info("Generating synthetic training data...")
            historical_data = self.data_generator.generate_historical_data(3000)
            
            logger.info("Training ML ETA prediction model...")
            accuracy = self.ml_predictor.train_model(historical_data)
            self.enhanced_metrics['ml_accuracy'] = accuracy
            
            logger.info(f"ML model initialized with accuracy: {accuracy:.3f}")
        except Exception as e:
            logger.error(f"Failed to initialize ML model: {e}")
            self.enhanced_metrics['ml_accuracy'] = 0

    async def add_websocket(self, websocket: WebSocket): 
        self.websocket_connections.add(websocket)
        
    async def remove_websocket(self, websocket: WebSocket): 
        self.websocket_connections.discard(websocket)

    async def broadcast_state(self):
        if not self.websocket_connections: 
            return
            
        try:
            state = self.get_system_state()
            state_json = json.dumps(state, default=str)  # Add default=str to handle datetime objects
            disconnected_clients = set()
            
            for websocket in self.websocket_connections:
                try: 
                    await websocket.send_text(state_json)
                except Exception as e: 
                    logger.warning(f"Failed to send to websocket: {e}")
                    disconnected_clients.add(websocket)
                    
            for client in disconnected_clients: 
                self.websocket_connections.discard(client)
        except Exception as e:
            logger.error(f"Failed to broadcast state: {e}")

    def inject_delay(self, train_id: str, delay_minutes: int):
        """Inject artificial delay into a train"""
        try:
            if train_id in self.trains:
                train = self.trains[train_id]
                train['speed'] = max(10, train['speed'] - (delay_minutes * 2))
                train['injected_delay'] = delay_minutes
                
                # Create a proper event structure
                event_message = f"Delay Injected: {train['number']} delayed by {delay_minutes}min"
                self.events.append({
                    'type': 'delay_injection',
                    'details': f"{train['number']} delayed by {delay_minutes} minutes",
                    'timestamp': datetime.now(),
                    'train_id': train_id
                })
                
                logger.info(f"Injected {delay_minutes}min delay to train {train_id}")
                return True
            else:
                logger.warning(f"Train {train_id} not found for delay injection")
                return False
        except Exception as e:
            logger.error(f"Failed to inject delay: {e}")
            return False

    def apply_optimization_recommendation(self, recommendation):
        """Apply an optimization recommendation"""
        try:
            train_id = recommendation.get('train_id')
            if not train_id:
                logger.error("No train_id in recommendation")
                return False
                
            if train_id not in self.trains:
                logger.error(f"Train {train_id} not found in system")
                return False
                
            train = self.trains[train_id]
            logger.info(f"Applying recommendation to train {train_id}: {recommendation}")
            
            if recommendation['type'] == 'speed_adjustment':
                old_speed = train['speed']
                new_speed = recommendation.get('recommended_speed', old_speed)
                
                # Validate speed range
                if 30 <= new_speed <= 160:
                    train['speed'] = new_speed
                    self.events.append({
                        'type': 'speed_adjustment',
                        'details': f"{train['number']} speed adjusted: {old_speed}→{new_speed} km/h",
                        'timestamp': datetime.now(),
                        'train_id': train_id
                    })
                    logger.info(f"Speed adjusted for {train['number']}: {old_speed}→{new_speed}")
                else:
                    logger.warning(f"Invalid speed {new_speed} for train {train_id}")
                    return False
                
            elif recommendation['type'] == 'priority_adjustment':
                old_priority = train.get('priority', 99)
                new_priority = recommendation.get('recommended_priority', old_priority)
                
                # Validate priority range
                if 1 <= new_priority <= 99:
                    train['priority'] = new_priority
                    self.events.append({
                        'type': 'priority_adjustment',
                        'details': f"{train['number']} priority adjusted: P{old_priority}→P{new_priority}",
                        'timestamp': datetime.now(),
                        'train_id': train_id
                    })
                    logger.info(f"Priority adjusted for {train['number']}: P{old_priority}→P{new_priority}")
                else:
                    logger.warning(f"Invalid priority {new_priority} for train {train_id}")
                    return False
                    
            elif recommendation['type'] == 'reroute_suggestion':
                # For now, just log the reroute suggestion without implementing it
                self.events.append({
                    'type': 'reroute_suggestion',
                    'details': f"{train['number']} reroute suggested (manual intervention required)",
                    'timestamp': datetime.now(),
                    'train_id': train_id
                })
                logger.info(f"Reroute suggested for {train['number']}")
            else:
                logger.warning(f"Unknown recommendation type: {recommendation['type']}")
                return False
                
            # Log the successful application
            self.audit_logger.log_recommendation(recommendation, accepted=True)
            self.enhanced_metrics['recommendations_accepted'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimization recommendation: {e}", exc_info=True)
            return False

    def get_ml_predictions(self):
        """Get ML ETA predictions for all active trains"""
        predictions = {}
        
        try:
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
                            'confidence': 0.85  # Mock confidence score
                        }
        except Exception as e:
            logger.error(f"Failed to get ML predictions: {e}")
                    
        return predictions

    def get_optimization_recommendations(self):
        """Get current optimization recommendations"""
        try:
            recommendations = self.optimizer.optimize_schedule(
                self.trains, 
                self.ml_predictor, 
                GRAPH, 
                self.disrupted_sections, 
                self.disrupted_trains
            )
            
            for rec in recommendations:
                self.audit_logger.log_recommendation(rec, accepted=False)
                self.enhanced_metrics['total_recommendations'] += 1
                
            return recommendations
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return []

    def _update_enhanced_metrics(self):
        """Update enhanced KPIs including ML accuracy and on-time performance"""
        try:
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
        except Exception as e:
            logger.error(f"Failed to update enhanced metrics: {e}")

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
    
    def add_train(self, train_data: dict):
        try:
            train_id, start_stn, dest_stn = train_data['id'], f"STN_{train_data['start']}", f"STN_{train_data['destination']}"
            route = self.find_shortest_path(start_stn, dest_stn)
            if not route: 
                logger.warning(f"No route found from {start_stn} to {dest_stn}")
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
        except Exception as e:
            logger.error(f"Failed to add train: {e}")
            return None

    def _update_metrics(self):
        try:
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
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    async def update_simulation(self):
        try:
            self.events.clear()
            if self.is_running:
                self.simulation_time += 1
                await asyncio.gather(*(self._update_train(tid) for tid in list(self.trains.keys())))
                self._update_metrics()
                await self.broadcast_state()
        except Exception as e:
            logger.error(f"Failed to update simulation: {e}")

    async def _update_train(self, train_id: str):
        try:
            if train_id in self.disrupted_trains:
                return
            
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
                
                self.events.append({
                    'type': 'arrival',
                    'details': f"{train['number']} arrived at {final_section}",
                    'timestamp': datetime.now(),
                    'train_id': train_id
                })
                return
                
            # Skip if already completed
            if train['statusType'] == 'completed': 
                return
            
            train['statusType'] = 'running'
            current_section, next_section = train['route'][current_idx], train['route'][current_idx + 1]
            is_at_station, is_stop = current_section.startswith('STN_'), current_section in train.get('stops', [])
            
            if is_at_station and is_stop:
                if not train['atStation']: 
                    train['atStation'], train['dwellTimeStart'] = True, self.simulation_time
                    self.events.append({
                        'type': 'halt',
                        'details': f"{train['number']} halted at {current_section}",
                        'timestamp': datetime.now(),
                        'train_id': train_id
                    })
                if self.simulation_time - train['dwellTimeStart'] < 5: 
                    train['status'] = f"Halting at {current_section}"
                    return
                    
            if not current_section.startswith('STN_') and train['atStation']: 
                train['atStation'] = False
            
            required_time = self.calculate_travel_time(train['speed'], is_at_station and not is_stop)
            
            if self.simulation_time - progress['lastMoveTime'] >= required_time:
                if self.is_section_available(next_section, train_id):
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
                            self.events.append({
                                'type': 'conflict',
                                'details': f"{train['number']} waits for {occupying_train['number']}",
                                'timestamp': datetime.now(),
                                'train_id': train_id
                            })
        except Exception as e:
            logger.error(f"Failed to update train {train_id}: {e}")

    def get_system_state(self): 
        try:
            # Format events properly
            formatted_events = []
            for event in self.events[-10:]:  # Last 10 events
                if isinstance(event, dict):
                    formatted_events.append({
                        'type': event.get('type', 'unknown'),
                        'message': event.get('details', 'No details'),
                        'timestamp': event.get('timestamp', datetime.now()).strftime('%H:%M:%S') if isinstance(event.get('timestamp'), datetime) else str(event.get('timestamp', ''))
                    })
                else:
                    # Handle old string format
                    formatted_events.append({
                        'type': 'system',
                        'message': str(event),
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
            
            state = {
                "trains": list(self.trains.values()), 
                "blockOccupancy": self.block_occupancy, 
                "stationPlatforms": self.station_platforms, 
                "simulationTime": self.simulation_time, 
                "isRunning": self.is_running, 
                "trainProgress": self.train_progress, 
                "metrics": self.metrics, 
                "events": formatted_events,
                "enhancedMetrics": self.enhanced_metrics,
                "mlPredictions": self.get_ml_predictions(),
                "optimizationRecommendations": self.get_optimization_recommendations()
            }
            return state
        except Exception as e:
            logger.error(f"Failed to get system state: {e}")
            return {
                "trains": [], "blockOccupancy": {}, "stationPlatforms": {},
                "simulationTime": 0, "isRunning": False, "trainProgress": {},
                "metrics": {"throughput": 0, "avgDelay": 0, "utilization": 0, "avgSpeed": 0},
                "events": [], "enhancedMetrics": self.enhanced_metrics,
                "mlPredictions": {}, "optimizationRecommendations": []
            }

    def find_shortest_path(self, start_node: str, end_node: str, avoid_sections: Set[str] = None) -> List[str]:
        avoid_sections = avoid_sections or set()
        distances = {node: float('inf') for node in GRAPH if node not in avoid_sections}
        if start_node not in distances:
            return []
            
        distances[start_node] = 0
        pq = [(0, start_node)]
        prev_nodes = {node: None for node in distances}
        
        while pq:
            dist, current = heapq.heappop(pq)
            if dist > distances[current]: 
                continue
            if current == end_node: 
                break
                
            for neighbor, weight in GRAPH.get(current, {}).items():
                if neighbor in avoid_sections:
                    continue
                new_dist = dist + weight
                if new_dist < distances.get(neighbor, float('inf')):
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
        if section_id in self.disrupted_sections:
            return False
        if section_id in self.block_occupancy: 
            return self.block_occupancy[section_id] is None
        elif section_id in self.station_platforms: 
            return any(p is None for p in self.station_platforms[section_id].values())
        return False
        
    def reset_simulation(self):
        try:
            self.is_running, self.simulation_time = False, 0
            self.trains.clear()
            self.train_progress.clear()
            self.completed_train_stats.clear()
            self.events.clear()
            self.disrupted_sections.clear()
            self.disrupted_trains.clear()
            
            for sec_id in self.block_occupancy: 
                self.block_occupancy[sec_id] = None
            for stn_id in self.station_platforms:
                for p_num in self.station_platforms[stn_id]: 
                    self.station_platforms[stn_id][p_num] = None
                    
            # Reset metrics
            self.enhanced_metrics.update({
                'recommendations_accepted': 0,
                'total_recommendations': 0
            })
            
            add_default_trains(self)
            logger.info("Enhanced simulation reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset simulation: {e}")
        
    def start_simulation(self): 
        self.is_running = True
        logger.info("Simulation started")
        
    def pause_simulation(self): 
        self.is_running = False
        logger.info("Simulation paused")

    def inject_accident(self, section_id: str, duration: int):
        """Simulate accident: Halt all trains in section for duration."""
        try:
            if section_id in self.disrupted_sections:
                return False
                
            self.disrupted_sections.add(section_id)
            
            # Halt trains in section
            for train_id, train in self.trains.items():
                if train['section'] == section_id:
                    train['statusType'] = 'halted'
                    train['status'] = f"Halted due to accident in {section_id}"
                    self.events.append({
                        'type': 'accident_halt',
                        'details': f"{train['number']} halted due to accident in {section_id}",
                        'timestamp': datetime.now(),
                        'train_id': train_id
                    })
            
            self.events.append({
                'type': 'accident',
                'details': f"Accident in {section_id} for {duration} minutes",
                'timestamp': datetime.now(),
                'section_id': section_id
            })
            
            logger.info(f"Accident injected in {section_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to inject accident: {e}")
            return False

    def inject_breakdown(self, train_id: str, duration: int):
        """Simulate breakdown: Halt specific train for duration."""
        try:
            if train_id not in self.trains or train_id in self.disrupted_trains:
                return False
                
            self.disrupted_trains.add(train_id)
            train = self.trains[train_id]
            train['statusType'] = 'halted'
            train['status'] = f"Broken down for {duration} min"
            
            self.events.append({
                'type': 'breakdown',
                'details': f"{train['number']} breakdown for {duration} minutes",
                'timestamp': datetime.now(),
                'train_id': train_id
            })
            
            logger.info(f"Breakdown injected for train {train_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to inject breakdown: {e}")
            return False

    def inject_track_closure(self, section_id: str, duration: int):
        """Close track: Mark section unavailable, reroute trains."""
        try:
            if section_id in self.disrupted_sections:
                return False
                
            self.disrupted_sections.add(section_id)
            
            # Reroute affected trains
            rerouted_count = 0
            for train_id, train in self.trains.items():
                if section_id in train['route']:
                    new_route = self.find_shortest_path(
                        train['route'][0], 
                        f"STN_{train['destination']}", 
                        avoid_sections=self.disrupted_sections
                    )
                    if new_route:
                        train['route'] = new_route
                        rerouted_count += 1
                        self.events.append({
                            'type': 'reroute',
                            'details': f"{train['number']} rerouted due to closure in {section_id}",
                            'timestamp': datetime.now(),
                            'train_id': train_id
                        })
            
            self.events.append({
                'type': 'track_closure',
                'details': f"Track closed: {section_id} for {duration} min, {rerouted_count} trains rerouted",
                'timestamp': datetime.now(),
                'section_id': section_id
            })
            
            logger.info(f"Track closure in {section_id}, rerouted {rerouted_count} trains")
            return True
        except Exception as e:
            logger.error(f"Failed to inject track closure: {e}")
            return False

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
        {'id': 'T8', 'number': '42801', 'name': 'Cross Network', 'start': 'H', 'destination': 'L', 
         'speed': 100, 'departureTime': 14, 'stops': ['STN_H', 'STN_F', 'STN_B', 'STN_K', 'STN_L'], 'priority': 20},
        {'id': 'T9', 'number': '42802', 'name': 'Circle Line', 'start': 'A', 'destination': 'A', 
         'speed': 90, 'departureTime': 16, 'stops': ['STN_A', 'STN_E', 'STN_G', 'STN_C', 'STN_A'], 'priority': 35},
        
        # Peak hour services
        {'id': 'T10', 'number': '52901', 'name': 'Peak Express', 'start': 'G', 'destination': 'A', 
         'speed': 110, 'departureTime': 18, 'stops': ['STN_G', 'STN_C', 'STN_A'], 'priority': 8},
        {'id': 'T11', 'number': '52902', 'name': 'Commuter Rush', 'start': 'I', 'destination': 'J', 
         'speed': 95, 'departureTime': 20, 'stops': ['STN_I', 'STN_H', 'STN_F', 'STN_B', 'STN_J'], 'priority': 12},
        
        # Additional services for network testing
        {'id': 'T12', 'number': '62001', 'name': 'Network Test A', 'start': 'D', 'destination': 'E', 
         'speed': 85, 'departureTime': 22, 'stops': ['STN_D', 'STN_C', 'STN_G', 'STN_F', 'STN_E'], 'priority': 45},
    ]
    
    try:
        for train_data in default_trains: 
            system.add_train(train_data)
        logger.info(f"Added {len(default_trains)} default trains")
    except Exception as e:
        logger.error(f"Failed to add default trains: {e}")