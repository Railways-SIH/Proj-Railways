# backend/ml/predictor.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
# ------------------ ADD THIS LINE ------------------
from typing import List, Dict, Any, Optional 
# ---------------------------------------------------
import logging

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
    actual_travel_time: int # Time in Ticks
    delay: int # Delay in Ticks
    timestamp: datetime

class SyntheticDataGenerator:
    def __init__(self):
        self.weather_conditions = ['clear', 'rain', 'fog', 'snow', 'storm']
        self.historical_data: List[HistoricalRecord] = []
        
    def generate_historical_data(self, num_records=2000) -> List[HistoricalRecord]:
        """Generate synthetic historical train data for ML training"""
        data = []
        
        for i in range(num_records):
            route_length = random.randint(3, 15) 
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
        
    def prepare_features(self, data: List[HistoricalRecord]) -> pd.DataFrame:
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
        
    def train_model(self, historical_data: List[HistoricalRecord]) -> float:
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
        
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"ML Model trained - Test R²: {test_score:.3f}")
        
        self.is_trained = True
        return test_score
        
    def predict_eta(self, route_length: int, scheduled_speed: int, current_conditions: Dict[str, Any]) -> Optional[int]:
        """Predict ETA (in Ticks) for a train given current conditions"""
        if not self.is_trained:
            return None
        
        weather = current_conditions.get('weather', 'clear')
        features_data = {
            'route_length': route_length,
            'scheduled_speed': scheduled_speed,
            'actual_speed': scheduled_speed, 
            'weather_clear': 1 if weather == 'clear' else 0,
            'weather_rain': 1 if weather == 'rain' else 0,
            'weather_fog': 1 if weather == 'fog' else 0,
            'weather_snow': 1 if weather == 'snow' else 0,
            'weather_storm': 1 if weather == 'storm' else 0,
            'time_of_day': current_conditions.get('time_of_day', 12),
            'network_congestion': current_conditions.get('network_congestion', 0.5)
        }
        
        features_df = pd.DataFrame([features_data])
        features_scaled = self.scaler.transform(features_df)
        predicted_time = self.model.predict(features_scaled)[0]
        
        return max(route_length * 3, int(predicted_time))