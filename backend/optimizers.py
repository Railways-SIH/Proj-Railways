from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

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
            
            # Calculate base travel time
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
        try:
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
            
            # Calculate accuracy
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"ML Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
            self.is_trained = True
            return test_score
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")
            self.is_trained = False
            return 0.0
        
    def predict_eta(self, route_length, scheduled_speed, current_conditions):
        """Predict ETA for a train given current conditions"""
        try:
            if not self.is_trained or not self.model:
                return None
                
            # Use DataFrame with column names
            columns = ['route_length', 'scheduled_speed', 'actual_speed', 
                       'weather_clear', 'weather_rain', 'weather_fog', 'weather_snow', 'weather_storm',
                       'time_of_day', 'network_congestion']
            
            weather = current_conditions.get('weather', 'clear')
            features = pd.DataFrame([[
                route_length,
                scheduled_speed,
                scheduled_speed,  # assume actual speed = scheduled initially
                1 if weather == 'clear' else 0,
                1 if weather == 'rain' else 0,
                1 if weather == 'fog' else 0,
                1 if weather == 'snow' else 0,
                1 if weather == 'storm' else 0,
                current_conditions.get('time_of_day', 12),
                current_conditions.get('network_congestion', 0.5)
            ]], columns=columns)
            
            features_scaled = self.scaler.transform(features)
            predicted_time = self.model.predict(features_scaled)[0]
            
            return max(route_length * 3, int(predicted_time))  # minimum reasonable time
        except Exception as e:
            logger.error(f"Failed to predict ETA: {e}")
            return None

class ScheduleOptimizer:
    def __init__(self):
        self.current_conditions = {
            'weather': 'clear',
            'time_of_day': 12,
            'network_congestion': 0.3
        }
        
    def update_conditions(self, conditions):
        """Update current operating conditions"""
        try:
            self.current_conditions.update(conditions)
            logger.info(f"Updated conditions: {conditions}")
        except Exception as e:
            logger.error(f"Failed to update conditions: {e}")
        
    def optimize_schedule(self, trains, ml_predictor, graph, disrupted_sections=None, disrupted_trains=None):
        """Optimize train schedule using heuristic algorithms instead of OR-Tools"""
        recommendations = []
        disrupted_sections = disrupted_sections or set()
        disrupted_trains = disrupted_trains or set()
        
        try:
            # Filter active trains
            active_trains = {
                tid: train for tid, train in trains.items()
                if (train['statusType'] in ['running', 'scheduled'] and 
                    tid not in disrupted_trains)
            }
            
            logger.info(f"Analyzing {len(active_trains)} active trains for optimization")
            
            # Priority-based optimization (simple heuristic)
            train_priorities = []
            for train_id, train in active_trains.items():
                priority_score = self._calculate_priority_score(train, ml_predictor)
                train_priorities.append((priority_score, train_id, train))
            
            # Sort by priority (lower score = higher priority)
            train_priorities.sort()
            
            # Generate recommendations based on priority and predictions
            for priority_score, train_id, train in train_priorities[:8]:  # Limit to top 8 trains
                route_length = len(train.get('route', []))
                current_speed = train.get('speed', 80)
                
                # ML prediction integration
                predicted_delay = 0
                if ml_predictor.is_trained:
                    predicted_eta = ml_predictor.predict_eta(
                        route_length, 
                        current_speed, 
                        self.current_conditions
                    )
                    
                    if predicted_eta:
                        ideal_time = train.get('idealTravelTime', route_length * 5)
                        predicted_delay = max(0, predicted_eta - ideal_time)
                        logger.info(f"Train {train_id}: predicted delay = {predicted_delay}")
                
                # Speed adjustment recommendations
                if predicted_delay > 5 or train.get('waitingForBlock', False):
                    new_speed = min(160, int(current_speed * 1.15))
                    if new_speed > current_speed:
                        recommendations.append({
                            'type': 'speed_adjustment',
                            'train_id': train_id,
                            'train_number': train.get('number', train_id),
                            'current_speed': current_speed,
                            'recommended_speed': new_speed,
                            'predicted_delay': predicted_delay,
                            'reason': f'Predicted delay of {predicted_delay} ticks - increase speed to {new_speed} km/h',
                            'priority': train.get('priority', 99)
                        })
                        logger.info(f"Generated speed recommendation for {train_id}: {current_speed} -> {new_speed}")
                
                # Early arrival optimization
                elif predicted_delay < -2 and current_speed > 50:
                    new_speed = max(40, int(current_speed * 0.95))
                    if new_speed < current_speed:
                        recommendations.append({
                            'type': 'speed_adjustment',
                            'train_id': train_id,
                            'train_number': train.get('number', train_id),
                            'current_speed': current_speed,
                            'recommended_speed': new_speed,
                            'predicted_delay': predicted_delay,
                            'reason': f'Early arrival predicted - reduce speed to {new_speed} km/h for energy savings',
                            'priority': train.get('priority', 99)
                        })
                        logger.info(f"Generated energy optimization for {train_id}: {current_speed} -> {new_speed}")
                
                # Priority adjustment for delayed/waiting trains
                if (train.get('waitingForBlock', False) or predicted_delay > 3) and train.get('priority', 99) > 15:
                    current_priority = train.get('priority', 99)
                    new_priority = max(5, current_priority - 10)
                    recommendations.append({
                        'type': 'priority_adjustment',
                        'train_id': train_id,
                        'train_number': train.get('number', train_id),
                        'current_priority': current_priority,
                        'recommended_priority': new_priority,
                        'reason': f'Train experiencing delays - increase priority from P{current_priority} to P{new_priority}',
                        'predicted_delay': predicted_delay
                    })
                    logger.info(f"Generated priority recommendation for {train_id}: P{current_priority} -> P{new_priority}")
                
                # Stop generating once we have enough recommendations
                if len(recommendations) >= 3:
                    break
            
            # Add some general system recommendations if few train-specific ones
            if len(recommendations) < 2:
                waiting_count = sum(1 for t in active_trains.values() if t.get('waitingForBlock', False))
                if waiting_count > 2:
                    recommendations.append({
                        'type': 'system_recommendation',
                        'train_id': 'SYSTEM',
                        'train_number': 'SYSTEM',
                        'reason': f'{waiting_count} trains waiting - consider manual intervention or rerouting',
                        'predicted_delay': 0,
                        'current_priority': 50,
                        'recommended_priority': 40
                    })
                    
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to optimize schedule: {e}", exc_info=True)
            return []
    
    def _calculate_priority_score(self, train, ml_predictor):
        """Calculate priority score for train (lower = higher priority)"""
        try:
            base_priority = train.get('priority', 99)
            
            # Adjust for delays
            if train.get('waitingForBlock', False):
                base_priority -= 20
            
            # Adjust for predicted delays
            if ml_predictor.is_trained:
                route_length = len(train.get('route', []))
                predicted_eta = ml_predictor.predict_eta(
                    route_length, 
                    train['speed'], 
                    self.current_conditions
                )
                if predicted_eta:
                    ideal_time = train.get('idealTravelTime', route_length * 5)
                    predicted_delay = max(0, predicted_eta - ideal_time)
                    if predicted_delay > 5:
                        base_priority -= predicted_delay
            
            return base_priority
        except Exception as e:
            logger.error(f"Failed to calculate priority score: {e}")
            return 99
    
    def _resolve_conflicts(self, trains, disrupted_sections):
        """Generate recommendations to resolve conflicts"""
        recommendations = []
        
        try:
            # Find trains that might be in conflict
            section_occupancy = {}
            for train_id, train in trains.items():
                section = train.get('section')
                if section and section not in disrupted_sections:
                    if section not in section_occupancy:
                        section_occupancy[section] = []
                    section_occupancy[section].append((train_id, train))
            
            # Check for potential conflicts in upcoming sections
            waiting_trains = [
                (tid, train) for tid, train in trains.items() 
                if train.get('waitingForBlock', False)
            ]
            
            if waiting_trains:
                # Suggest rerouting for low-priority waiting trains
                for train_id, train in waiting_trains:
                    if train.get('priority', 99) > 30:  # Low priority trains
                        recommendations.append({
                            'type': 'reroute_suggestion',
                            'train_id': train_id,
                            'train_number': train['number'],
                            'current_priority': train.get('priority', 99),
                            'reason': 'Consider rerouting to reduce congestion',
                            'predicted_delay': 0
                        })
            
        except Exception as e:
            logger.error(f"Failed to resolve conflicts: {e}")
        
        return recommendations

class AuditLogger:
    def __init__(self):
        self.audit_log = []
        self.kpi_history = []
        
    def log_recommendation(self, recommendation, accepted=False):
        """Log optimization recommendations"""
        try:
            log_entry = {
                'timestamp': datetime.now(),
                'type': 'recommendation',
                'data': recommendation,
                'accepted': accepted,
                'train_id': recommendation.get('train_id', 'unknown'),
                'recommendation_type': recommendation.get('type', 'unknown')
            }
            self.audit_log.append(log_entry)
            
            # Keep only last 1000 entries
            if len(self.audit_log) > 1000:
                self.audit_log = self.audit_log[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to log recommendation: {e}")
        
    def log_kpi(self, kpis):
        """Log KPI snapshots"""
        try:
            kpi_entry = {
                'timestamp': datetime.now(),
                'kpis': kpis.copy()  # Make a copy to avoid reference issues
            }
            self.kpi_history.append(kpi_entry)
            
            # Keep only last 24 hours of data (assuming 1 entry per minute)
            if len(self.kpi_history) > 1440:
                self.kpi_history = self.kpi_history[-1440:]
                
        except Exception as e:
            logger.error(f"Failed to log KPI: {e}")
        
    def get_recent_logs(self, limit=50):
        """Get recent audit logs"""
        try:
            recent_logs = self.audit_log[-limit:] if self.audit_log else []
            
            # Format logs for API response
            formatted_logs = []
            for log in recent_logs:
                try:
                    formatted_log = {
                        'timestamp': log['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(log['timestamp'], datetime) else str(log['timestamp']),
                        'type': log.get('type', 'unknown'),
                        'accepted': log.get('accepted', False),
                        'train_id': log.get('train_id', 'unknown'),
                        'recommendation_type': log.get('recommendation_type', 'unknown'),
                        'details': str(log.get('data', {}))
                    }
                    formatted_logs.append(formatted_log)
                except Exception as e:
                    logger.warning(f"Failed to format log entry: {e}")
                    continue
            
            return formatted_logs
        except Exception as e:
            logger.error(f"Failed to get recent logs: {e}")
            return []
        
    def get_kpi_history(self, hours=24):
        """Get KPI history for specified hours"""
        try:
            if not self.kpi_history:
                return []
                
            cutoff = datetime.now() - timedelta(hours=hours)
            filtered_history = [
                entry for entry in self.kpi_history 
                if entry['timestamp'] > cutoff
            ]
            
            # Format for API response
            formatted_history = []
            for entry in filtered_history:
                try:
                    formatted_entry = {
                        'timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry['timestamp'], datetime) else str(entry['timestamp']),
                        'kpis': entry.get('kpis', {})
                    }
                    formatted_history.append(formatted_entry)
                except Exception as e:
                    logger.warning(f"Failed to format KPI entry: {e}")
                    continue
            
            return formatted_history
        except Exception as e:
            logger.error(f"Failed to get KPI history: {e}")
            return []