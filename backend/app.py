from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Set, Tuple, Optional
import heapq
import uvicorn
import asyncio
import json
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

# --- CONFIGURATION CONSTANTS ---
# Define the core conversion factor: 1 tick is equivalent to 6 seconds (0.1 minutes)
# This makes 1 minute equal to 10 ticks (a clean number for simulation logic)
MINUTES_PER_TICK = 0.1
SECONDS_PER_TICK = 6
# -------------------------------

# --- UTILITY FUNCTION FOR TIME CONVERSION ---
def _convert_ticks_to_minutes(ticks: int) -> float:
    """Converts time from Ticks to Minutes."""
    return round(ticks * MINUTES_PER_TICK, 2)
# --------------------------------------------


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Enhanced Intelligent Railway Control Backend", version="6.0.1") # Bumped version

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    {'id': 'STN_D', 'type': 'station', 'name': 'Terminal D', 'station': 'D', 'platforms': 3},
    
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
    {'id': 'STN_L', 'type': 'station', 'name': 'Harbor L', 'station': 'L', 'platforms': 3},
    
    # Junction blocks
    {'id': 'BLOCK_V_A_E', 'type': 'block', 'name': 'V-Block (A-E)'},
    {'id': 'BLOCK_V_A_J', 'type': 'block', 'name': 'V-Block (A-J)'},
    {'id': 'BLOCK_V_B_F', 'type': 'block', 'name': 'V-Block (B-F)'},
    {'id': 'BLOCK_V_F_H', 'type': 'block', 'name': 'V-Block (F-H)'},
    {'id': 'BLOCK_V_B_K', 'type': 'block', 'name': 'V-Block (B-K)'},
    {'id': 'BLOCK_V_C_G', 'type': 'block', 'name': 'V-Block (C-G)'},
]

# CORRECTED GRAPH with proper connections
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
    
    # Upper branch connections (FIXED - was missing connection between BLOCK_H2 and STN_I)
    'STN_H': {'BLOCK_H1': 5, 'BLOCK_V_F_H': 4}, 
    'BLOCK_H1': {'STN_H': 5, 'BLOCK_H2': 5},
    'BLOCK_H2': {'BLOCK_H1': 5, 'STN_I': 5},  # FIXED: removed duplicate 'BLOCK_H2': 5
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
    actual_travel_time: int # Time in Ticks
    delay: int # Delay in Ticks
    timestamp: datetime

class SyntheticDataGenerator:
    def __init__(self):
        self.weather_conditions = ['clear', 'rain', 'fog', 'snow', 'storm']
        self.historical_data = []
        
    def generate_historical_data(self, num_records=2000):
        """Generate synthetic historical train data for ML training"""
        data = []
        
        for i in range(num_records):
            route_length = random.randint(3, 15)  # Increased for larger network
            scheduled_speed = random.choice([40, 60, 80, 100, 120, 140, 160])
            weather = random.choice(self.weather_conditions)
            time_of_day = random.randint(0, 23)
            network_congestion = random.uniform(0.1, 0.9)
            
            # Calculate base travel time (in Ticks)
            base_travel_time = route_length * 5
            
            # Apply factors for actual travel time
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
        
        # Calculate accuracy (R-squared)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"ML Model trained - Test R²: {test_score:.3f}")
        
        self.is_trained = True
        return test_score
        
    def predict_eta(self, route_length, scheduled_speed, current_conditions):
        """Predict ETA (in Ticks) for a train given current conditions"""
        if not self.is_trained:
            return None
        
        # Create DataFrame with proper feature names to match training data
        weather = current_conditions.get('weather', 'clear')
        features_data = {
            'route_length': route_length,
            'scheduled_speed': scheduled_speed,
            'actual_speed': scheduled_speed,  # assume actual speed = scheduled initially
            'weather_clear': 1 if weather == 'clear' else 0,
            'weather_rain': 1 if weather == 'rain' else 0,
            'weather_fog': 1 if weather == 'fog' else 0,
            'weather_snow': 1 if weather == 'snow' else 0,
            'weather_storm': 1 if weather == 'storm' else 0,
            'time_of_day': current_conditions.get('time_of_day', 12),
            'network_congestion': current_conditions.get('network_congestion', 0.5)
        }
        
        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([features_data])
        features_scaled = self.scaler.transform(features_df)
        predicted_time = self.model.predict(features_scaled)[0]
        
        return max(route_length * 3, int(predicted_time))  # minimum reasonable time in Ticks

class ScheduleOptimizer:
    def __init__(self):
        self.current_conditions = {
            'weather': 'clear',
            'time_of_day': 12,
            'network_congestion': 0.3
        }
        
    def update_conditions(self, conditions):
        self.current_conditions.update(conditions)
        
    def optimize_schedule(self, trains, ml_predictor):
        """Optimize train schedule using ML predictions and conflict resolution"""
        recommendations = []
        
        for train_id, train in trains.items():
            if train['statusType'] in ['completed', 'cancelled']:
                continue
                
            route_length = len(train.get('route', []))
            
            # Get ML prediction
            if ml_predictor.is_trained:
                predicted_eta_ticks = ml_predictor.predict_eta(
                    route_length, 
                    train['speed'], 
                    self.current_conditions
                )
                
                # Compare with ideal time (in Ticks)
                ideal_time_ticks = train.get('idealTravelTime', route_length * 5)
                
                # Convert delay to minutes for display/logic
                predicted_delay_ticks = predicted_eta_ticks - ideal_time_ticks
                predicted_delay_min = _convert_ticks_to_minutes(predicted_delay_ticks) # Use utility function
                
                if predicted_delay_min > 0.5:  # Significant delay predicted (30 seconds)
                    recommendations.append({
                        'type': 'speed_adjustment',
                        'train_id': train_id,
                        'train_number': train['number'],
                        'current_speed': train['speed'],
                        'recommended_speed': min(160, int(train['speed'] * 1.2)),
                        'predicted_delay': round(predicted_delay_min, 1),
                        'reason': f'Predicted delay of {round(predicted_delay_min, 1)} min. Increase speed to catch up.'
                    })
                    
                elif predicted_delay_min < -0.5:  # Early arrival
                    recommendations.append({
                        'type': 'speed_adjustment',
                        'train_id': train_id,
                        'train_number': train['number'],
                        'current_speed': train['speed'],
                        'recommended_speed': max(30, int(train['speed'] * 0.9)),
                        'predicted_delay': round(predicted_delay_min, 1),
                        'reason': f'Predicted early arrival by {round(abs(predicted_delay_min), 1)} min. Reduce speed to maintain schedule.'
                    })
                    
            # Priority-based recommendations
            if train['waitingForBlock']:
                recommendations.append({
                    'type': 'priority_adjustment',
                    'train_id': train_id,
                    'train_number': train['number'],
                    'current_priority': train.get('priority', 99),
                    'recommended_priority': max(1, train.get('priority', 99) - 5),
                    'reason': 'Train currently blocked. Recommend temporary priority increase.'
                })
                
        return recommendations

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
        self.simulation_time = 0 # In Ticks
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

    def inject_delay(self, train_id: str, delay_minutes: int):
        """Inject artificial delay into a train (speed reduction is proportional to delay)"""
        if train_id in self.trains:
            train = self.trains[train_id]
            
            # Convert delay minutes into a proportional speed reduction
            speed_reduction = delay_minutes * 3
            
            train['speed'] = max(10, train['speed'] - speed_reduction)
            train['injected_delay'] = delay_minutes
            
            self.events.append(f"Delay Injected: {train['number']} slowed, {delay_minutes} min delay simulated.")
            return True
        return False

    def apply_optimization_recommendation(self, recommendation):
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
        # Base confidence starts high, drops with delay
        # Max delay (e.g., 50 ticks) drops confidence to minimum 0.5
        max_delay_for_confidence = 50
        
        if predicted_delay_ticks <= 0:
            return 0.95
        
        # Linear decay: 0.95 - (delay / max_delay) * 0.45
        decay = (min(predicted_delay_ticks, max_delay_for_confidence) / max_delay_for_confidence) * 0.45
        confidence = max(0.5, 0.95 - decay)
        
        return round(confidence, 2)

    def get_ml_predictions(self):
        """Get ML ETA predictions (in minutes) for all active trains"""
        predictions = {}
        
        for train_id, train in self.trains.items():
            if train['statusType'] in ['running', 'scheduled']:
                # Calculate remaining route length in sections
                current_idx = self.train_progress.get(train_id, {}).get('currentRouteIndex', 0)
                remaining_route_length = len(train.get('route', [])) - current_idx
                
                if remaining_route_length <= 0:
                    continue
                
                # Predict ETA for the remaining route (in Ticks)
                predicted_eta_ticks = self.ml_predictor.predict_eta(
                    remaining_route_length,
                    train['speed'],
                    self.optimizer.current_conditions
                )
                
                if predicted_eta_ticks:
                    # Calculate remaining ideal time (in Ticks)
                    # Recalculate ideal time for remaining route only
                    remaining_route = train['route'][current_idx:]
                    remaining_stops = [s for s in train.get('stops', []) if s in remaining_route]
                    
                    ideal_time_ticks = self.calculate_ideal_travel_time(
                        remaining_route, train['speed'], remaining_stops
                    )
                    
                    predicted_delay_ticks = predicted_eta_ticks - ideal_time_ticks
                    
                    predictions[train_id] = {
                        # CONVERT TO MINUTES FOR FRONTEND DISPLAY using the utility function
                        'predicted_eta': _convert_ticks_to_minutes(predicted_eta_ticks),
                        'ideal_time': _convert_ticks_to_minutes(ideal_time_ticks),
                        'predicted_delay': _convert_ticks_to_minutes(predicted_delay_ticks),
                        # DYNAMIC CONFIDENCE
                        'confidence': self._calculate_confidence(predicted_delay_ticks)
                    }
                    
        return predictions

    def get_optimization_recommendations(self):
        """Get current optimization recommendations"""
        recommendations = self.optimizer.optimize_schedule(self.trains, self.ml_predictor)
        
        for rec in recommendations:
            # Only log new, unique recommendations
            # In a real system, you'd check uniqueness based on train_id and rec_type
            self.audit_logger.log_recommendation(rec, accepted=False)
            self.enhanced_metrics['total_recommendations'] += 1
            
        return recommendations

    def _update_enhanced_metrics(self):
        """Update enhanced KPIs including ML accuracy and on-time performance"""
        # Calculate on-time percentage
        if self.completed_train_stats:
            # On-time is defined as delay of 3 ticks (18 seconds/0.3 minutes) or less
            on_time_trains = sum(1 for s in self.completed_train_stats 
                                 if s['delay_ticks'] <= 3) # Use the delay_ticks stored
            self.enhanced_metrics['on_time_percentage'] = (on_time_trains / len(self.completed_train_stats)) * 100
        else:
            self.enhanced_metrics['on_time_percentage'] = 100
            
        # Log KPIs
        combined_metrics = {**self.metrics, **self.enhanced_metrics}
        self.audit_logger.log_kpi(combined_metrics)

    # Existing methods remain the same but with enhanced functionality
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
        # Returns time in Ticks
        if is_station_pass_through: 
            return 1
        speed_factor = max(0.5, min(2.0, 100 / train_speed))
        return max(3, int(5 * speed_factor))

    def calculate_ideal_travel_time(self, route: List[str], speed: int, stops: List[str]) -> int:
        # Returns time in Ticks
        ideal_time = 0
        for section_id in route:
            is_station, is_stop = section_id.startswith('STN_'), section_id in stops
            ideal_time += self.calculate_travel_time(speed, is_station_pass_through=(is_station and not is_stop))
            if is_stop: 
                ideal_time += 5 # 5 ticks for dwelling at a stop
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
            'idealTravelTime': ideal_time, 'priority': train_data.get('priority', 99), # idealTime is in Ticks
            'injected_delay': 0  # Track artificial delays (in Minutes)
        }
        self.trains[train_id] = train
        self.train_progress[train_id] = {'currentRouteIndex': 0, 'lastMoveTime': train_data.get('departureTime', 0)}
        self.occupy_section(route[0], train_id)
        return train

    def _update_metrics(self):
        completed_count = len(self.completed_train_stats)
        
        # Metrics calculated in ticks first, then converted to minutes for display
        simulation_time_min = _convert_ticks_to_minutes(self.simulation_time)
        
        self.metrics["throughput"] = (completed_count / simulation_time_min) * 60 if simulation_time_min > 0 else 0
        
        # Now calculating avgDelay using the converted minutes from the stats
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
        
        # Update enhanced metrics
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
        
        # Don't update if train hasn't departed yet
        if self.simulation_time < train['departureTime']: 
            return
        
        current_idx = progress['currentRouteIndex']
        route = train['route']
        
        # Check if train has completed its journey (reached final destination)
        if current_idx >= len(route) - 1 and train['statusType'] != 'completed':
            # Release the final section
            final_section = route[-1]
            self.release_section(final_section, train_id)
            
            actual_time_ticks = self.simulation_time - train['departureTime']
            ideal_time_ticks = train['idealTravelTime']
            delay_ticks = actual_time_ticks - ideal_time_ticks

            train.update({'status': 'Arrived', 'statusType': 'completed'})
            self.completed_train_stats.append({
                'id': train_id, 
                'ideal_time_ticks': ideal_time_ticks,
                'actual_time_ticks': actual_time_ticks,
                'delay_ticks': delay_ticks,
                'ideal_time_min': _convert_ticks_to_minutes(ideal_time_ticks), # Storing in minutes too
                'actual_time_min': _convert_ticks_to_minutes(actual_time_ticks), # Storing in minutes too
                'delay_min': _convert_ticks_to_minutes(delay_ticks) # Storing in minutes too
            })
            delay_min_display = _convert_ticks_to_minutes(delay_ticks)
            self.events.append(f"Arrival: {train['number']} at {final_section}. Delay: {delay_min_display:.2f} min.")
            return
            
        # Skip if already completed
        if train['statusType'] == 'completed': 
            return
        
        train['statusType'] = 'running'
        current_section = train['route'][current_idx]
        next_section = train['route'][current_idx + 1]
        is_at_station, is_stop = current_section.startswith('STN_'), current_section in train.get('stops', [])
        
        # --- Station Halting Logic ---
        if is_at_station and is_stop:
            if not train['atStation']: 
                train['atStation'], train['dwellTimeStart'] = True, self.simulation_time
                self.events.append(f"Halt: {train['number']} at {current_section}.")
            # 5 ticks dwell time
            if self.simulation_time - train['dwellTimeStart'] < 5: 
                train['status'] = f"Halting at {current_section}"
                return
        if not current_section.startswith('STN_') and train['atStation']: 
            train['atStation'] = False
        # -----------------------------
        
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

    def get_system_state(self): 
        state = {
            "trains": list(self.trains.values()), 
            "blockOccupancy": self.block_occupancy, 
            "stationPlatforms": self.station_platforms, 
            "simulationTime": self.simulation_time, 
            "simulationTimeMinutes": _convert_ticks_to_minutes(self.simulation_time), # New field for minutes
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
        {'id': 'T12', 'number': '62001', 'name': 'Network Test A', 'start': 'D', 'destination': 'D', 
         'speed': 85, 'departureTime': 22, 'stops': ['STN_D', 'STN_C', 'STN_G', 'STN_F', 'STN_D'], 'priority': 45},
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
        await asyncio.sleep(1.5)  # Slightly faster for more trains

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
    # In a real system, you'd store recommendations with IDs
    # For now, we'll get current recommendations and find by ID
    recommendations = traffic_system.get_optimization_recommendations()
    
    # Mock finding recommendation by ID (in real system, store with unique IDs)
    if opt_request.accept:
        for rec in recommendations[:1]:  # Apply first recommendation as example
            success = traffic_system.apply_optimization_recommendation(rec)
            if success:
                await traffic_system.broadcast_state()
                return {"status": "success", "message": "Optimization applied"}
    
    return {"status": "success", "message": "Optimization rejected"}

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

@app.get("/kpi-history")
async def get_kpi_history(hours: int = 24):
    """Get KPI history"""
    history = traffic_system.audit_logger.get_kpi_history(hours)
    return {"history": history}

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

@app.get("/station-status")
async def get_station_status():
    """Get detailed station occupancy status"""
    stations = [s for s in TRACK_SECTIONS if s['type'] == 'station']
    station_status = {}
    
    for station in stations:
        platforms = traffic_system.station_platforms.get(station['id'], {})
        occupied = sum(1 for occupant in platforms.values() if occupant is not None)
        total = len(platforms)
        
        platform_details = {}
        for p_num, occupant in platforms.items():
            occupant_details = None
            if occupant and occupant in traffic_system.trains:
                train = traffic_system.trains[occupant]
                # Add delay info to platform details
                delay_min = _convert_ticks_to_minutes(train.get('idealTravelTime', 0)) - _convert_ticks_to_minutes(traffic_system.simulation_time - train['departureTime'])
                occupant_details = {
                    'train_id': occupant,
                    'train_number': train.get('number', 'N/A'),
                    'status': train.get('status', 'N/A'),
                    'delay_min': round(delay_min, 2)
                }

            platform_details[p_num] = occupant_details
        
        station_status[station['id']] = {
            'name': station['name'],
            'station_code': station['station'],
            'platforms': platform_details, # Updated to use the new platform_details
            'occupied_platforms': occupied,
            'total_platforms': total,
            'occupancy_percentage': (occupied / total * 100) if total > 0 else 0,
            'status': 'full' if occupied == total else 'partial' if occupied > 0 else 'free'
        }
    
    return {"station_status": station_status}

@app.get("/network-overview")
async def get_network_overview():
    """Get comprehensive network overview"""
    total_trains = len(traffic_system.trains)
    running_trains = sum(1 for t in traffic_system.trains.values() if t['statusType'] == 'running')
    waiting_trains = sum(1 for t in traffic_system.trains.values() if t['waitingForBlock'])
    completed_trains = sum(1 for t in traffic_system.trains.values() if t['statusType'] == 'completed')
    
    total_blocks = len(traffic_system.block_occupancy)
    occupied_blocks = sum(1 for occupant in traffic_system.block_occupancy.values() if occupant is not None)
    
    total_platforms = sum(len(platforms) for platforms in traffic_system.station_platforms.values())
    occupied_platforms = sum(
        sum(1 for occupant in platforms.values() if occupant is not None)
        for platforms in traffic_system.station_platforms.values()
    )
    
    # Calculate formatted time using the utility function
    sim_time_minutes = _convert_ticks_to_minutes(traffic_system.simulation_time)
    
    return {
        "network_summary": {
            "total_trains": total_trains,
            "running_trains": running_trains,
            "waiting_trains": waiting_trains,
            "completed_trains": completed_trains,
            "total_blocks": total_blocks,
            "occupied_blocks": occupied_blocks,
            "free_blocks": total_blocks - occupied_blocks,
            "total_platforms": total_platforms,
            "occupied_platforms": occupied_platforms,
            "free_platforms": total_platforms - occupied_platforms,
            "network_utilization": ((occupied_blocks + occupied_platforms) / (total_blocks + total_platforms) * 100) if (total_blocks + total_platforms) > 0 else 0
        },
        "simulation_status": {
            "is_running": traffic_system.is_running,
            "simulation_time": traffic_system.simulation_time, # Keep ticks for raw data
            "simulation_time_minutes": sim_time_minutes, # New field for minutes
            # Formatted string for easier display
            "simulation_time_formatted": f"{math.floor(sim_time_minutes):02d}m:{int((sim_time_minutes * 60) % 60):02d}s"
        }
    }

if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8000)