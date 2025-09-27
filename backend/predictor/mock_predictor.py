import random
from typing import List, Dict, Any

def run_predictive_analysis(trains: List[Dict[str, Any]], optimal_schedule: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mocks an AI model that predicts the downstream impact of the OR schedule 
    and suggests proactive control actions.
    
    Since we lack real ML training, this uses simple heuristics based on OR output.
    """
    
    predictions = {}
    
    for train in trains:
        train_id = train['id']
        current_delay = train.get('delay', 0)
        
        # 1. Use OR schedule to predict delay propagation
        schedule_decision = optimal_schedule.get(train_id)
        
        predicted_delay_min = current_delay
        proactive_action = "Maintain Speed"
        
        if schedule_decision and schedule_decision['action'] == "HOLD":
            wait_min = round(schedule_decision['wait_time_seconds'] / 60, 1)
            
            # Predict downstream impact: 80% of current hold time propagates as delay
            predicted_delay_min += wait_min * 0.8
            
            # AI/RL Recommendation: If hold time > 5 minutes, suggest slowing down early.
            if wait_min > 5:
                proactive_action = f"Apply 30% speed reduction {schedule_decision['target_block']} before hold point to save energy."
            
        elif current_delay > 10:
            # If currently delayed but not holding, predict further slight delay due to congestion.
            predicted_delay_min += 2
            proactive_action = "Expedite: Ensure clear exit from current block."
            
        
        predictions[train_id] = {
            "predicted_total_delay_min": round(predicted_delay_min, 1),
            "next_proactive_action": proactive_action,
            "confidence": "High" if schedule_decision and schedule_decision['action'] == "HOLD" else "Medium",
        }
        
    return predictions
