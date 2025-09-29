# backend/api/apiendpoints.py (ADJUSTED INDENTATION - FINAL)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, APIRouter
from typing import Dict, Any, Optional, List
import json
import math
import logging
logger = logging.getLogger(__name__)

# Import components from your backend structure
from backend.core.system import EnhancedTrafficControlSystem
from backend.models.schemas import (
    TrainData, SimulationControl, DelayInjection, 
    OptimizationRequest, ConditionUpdate
)
from backend.core.const import _convert_ticks_to_minutes, TRACK_SECTIONS

# Use a separate router or a function to register endpoints on the main app
def register_api_endpoints(app: FastAPI, traffic_system: EnhancedTrafficControlSystem):
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        await traffic_system.add_websocket(websocket)
        try:
            # Send initial state immediately
            await websocket.send_text(json.dumps(traffic_system.get_system_state()))
            while True: 
                # Keep connection alive, listen for client messages (if needed)
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
            raise HTTPException(status_code=400, detail="Could not create train (route not found or invalid data)")
        await traffic_system.broadcast_state()
        return {"status": "success", "train": new_train}

    # --- Enhanced ML and Optimization endpoints ---

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
        """Apply or reject optimization recommendations"""
        if not traffic_system:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        try:
            # NOTE: get_optimization_recommendations() re-calculates recommendations and logs them.
            recommendations = traffic_system.get_optimization_recommendations()
            
            # --- LOGIC TO FIND RECOMMENDATION BY ID (INDEX) ---
            try:
                rec_index = int(opt_request.recommendation_id)
                
                if 0 <= rec_index < len(recommendations):
                    recommendation = recommendations[rec_index]
                    
                    if opt_request.accept:
                        # ... (Accept logic calls apply_optimization_recommendation which adds the ALERT event) ...
                        success = traffic_system.apply_optimization_recommendation(recommendation) 
                        
                        if success:
                            await traffic_system.broadcast_state()
                            return {"status": "success", "message": "Optimization applied successfully"}
                        else:
                            # This should only happen if the train unexpectedly disappeared or a check failed.
                            raise HTTPException(status_code=400, detail="Failed to apply optimization: Core system rejected change.")
                    
                    else:
                        # 🎯 REJECTION LOGIC
                        train_num = recommendation.get('train_number', 'N/A')
                        rec_type = recommendation.get('type', 'Change')
                        
                        traffic_system.audit_logger.log_recommendation(recommendation, accepted=False)
                        traffic_system.events.append(f"INFO: Rejected {rec_type} for {train_num}. Reverting to manual control.")
                        
                        await traffic_system.broadcast_state() # Broadcast the rejection event
                        return {"status": "success", "message": "Optimization rejected"}
                
                else:
                    raise HTTPException(status_code=404, detail="Recommendation not found in current list.")
            
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid recommendation ID format (must be integer).")
                
        except HTTPException:
            # Re-raise explicit HTTP exceptions
            raise
        except Exception as e:
            # Catch unexpected errors and log them
            logger.error(f"Failed to process optimization request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
    
    @app.get("/weather-status")
    async def get_weather_status() -> Dict[str, str]:
        """Get current weather conditions (independent API call)"""
        weather = traffic_system.optimizer.get_current_weather()
        return {"current_weather": weather}

    # --- AUDIT & KPI ENDPOINTS ---
    
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
                    actual_elapsed = traffic_system.simulation_time - train.get('departureTime', 0)
                    delay_ticks = actual_elapsed - train.get('idealTravelTime', 0)
                    delay_min = _convert_ticks_to_minutes(delay_ticks)
                    
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
                'platforms': platform_details,
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
                "simulation_time": traffic_system.simulation_time,
                "simulation_time_minutes": sim_time_minutes, 
                "simulation_time_formatted": f"{math.floor(sim_time_minutes):02d}m:{int((sim_time_minutes * 60) % 60):02d}s"
            }
        }