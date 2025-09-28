# backend/ml/optimizer.py

from typing import Dict, Any, List
from backend.ml.predictor import MLETAPredictor
from backend.core.const import _convert_ticks_to_minutes

class ScheduleOptimizer:
    def __init__(self):
        self.current_conditions = {
            'weather': 'clear',
            'time_of_day': 12,
            'network_congestion': 0.3
        }
        
    def update_conditions(self, conditions: Dict[str, Any]):
        self.current_conditions.update(conditions)
        
    def optimize_schedule(self, trains: Dict[str, Any], ml_predictor: MLETAPredictor) -> List[Dict[str, Any]]:
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
                
                if predicted_eta_ticks is None:
                    continue
                
                # Compare with ideal time (in Ticks)
                ideal_time_ticks = train.get('idealTravelTime', route_length * 5)
                
                predicted_delay_ticks = predicted_eta_ticks - ideal_time_ticks
                predicted_delay_min = _convert_ticks_to_minutes(predicted_delay_ticks)
                
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