# main.py - Fixed with proper error handling and logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import json
import logging
from core_simulation import EnhancedTrafficControlSystem, TRACK_SECTIONS, GRAPH, add_default_trains
from optimizers import AuditLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('railway_control.log')
    ]
)
logger = logging.getLogger(__name__)

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

class AccidentInjection(BaseModel):
    section_id: str
    duration: int

class BreakdownInjection(BaseModel):
    train_id: str
    duration: int

class TrackClosureInjection(BaseModel):
    section_id: str
    duration: int

class ScenarioSimulation(BaseModel):
    conditions: dict

# Global traffic system instance
traffic_system = None

async def simulation_loop():
    """Main simulation loop"""
    while True:
        try:
            if traffic_system and traffic_system.is_running: 
                await traffic_system.update_simulation()
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
        await asyncio.sleep(1.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global traffic_system
    try:
        # Initialize the enhanced system
        logger.info("Initializing Enhanced Traffic Control System...")
        traffic_system = EnhancedTrafficControlSystem()
        
        # Add default trains
        logger.info("Adding default trains...")
        add_default_trains(traffic_system)
        
        # Start simulation loop
        logger.info("Starting simulation loop...")
        asyncio.create_task(simulation_loop())
        
        logger.info("System initialization complete")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    finally:
        logger.info("Shutting down system...")

# Create FastAPI app with lifespan
app = FastAPI(
    lifespan=lifespan, 
    title="Enhanced Intelligent Railway Control Backend", 
    version="6.0.1",
    description="AI-Powered Railway Traffic Management System"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    if not traffic_system:
        await websocket.close(code=1000, reason="System not initialized")
        return
    
    await traffic_system.add_websocket(websocket)
    try:
        # Send initial state
        initial_state = traffic_system.get_system_state()
        await websocket.send_text(json.dumps(initial_state, default=str))
        
        # Keep connection alive
        while True: 
            await websocket.receive_text()
    except WebSocketDisconnect: 
        logger.info("WebSocket connection closed")
        await traffic_system.remove_websocket(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await traffic_system.remove_websocket(websocket)

@app.post("/simulation-control")
async def control_simulation(control: SimulationControl):
    """Control simulation state"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        action = control.action.lower()
        logger.info(f"Simulation control action: {action}")
        
        if action == "start": 
            traffic_system.start_simulation()
        elif action == "pause": 
            traffic_system.pause_simulation()
        elif action == "reset": 
            traffic_system.reset_simulation()
        else: 
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
            
        await traffic_system.broadcast_state()
        return {"status": "success", "action": action, "message": f"Simulation {action} successful"}
    except Exception as e:
        logger.error(f"Failed to control simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to {action} simulation: {str(e)}")

@app.post("/add-train")
async def add_train_endpoint(train: TrainData):
    """Add a new train to the system"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        logger.info(f"Adding train: {train.id}")
        new_train = traffic_system.add_train(train.dict())
        if new_train is None: 
            raise HTTPException(status_code=400, detail="Could not create train - no valid route found")
            
        await traffic_system.broadcast_state()
        return {"status": "success", "train": new_train, "message": "Train added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add train: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add train: {str(e)}")

@app.post("/inject-delay")
async def inject_delay(delay_data: DelayInjection):
    """Inject delay into a specific train"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        logger.info(f"Injecting delay: {delay_data.train_id}, {delay_data.delay_minutes}min")
        success = traffic_system.inject_delay(delay_data.train_id, delay_data.delay_minutes)
        if not success:
            raise HTTPException(status_code=404, detail="Train not found")
        
        await traffic_system.broadcast_state()
        return {
            "status": "success", 
            "message": f"Delay of {delay_data.delay_minutes} minutes injected into train {delay_data.train_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject delay: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to inject delay: {str(e)}")

@app.post("/apply-optimization")
async def apply_optimization(opt_request: OptimizationRequest):
    """Apply or reject optimization recommendations"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        recommendations = traffic_system.get_optimization_recommendations()
        
        # Find the recommendation by ID (using index for now)
        try:
            rec_index = int(opt_request.recommendation_id)
            if 0 <= rec_index < len(recommendations):
                recommendation = recommendations[rec_index]
                
                if opt_request.accept:
                    success = traffic_system.apply_optimization_recommendation(recommendation)
                    if success:
                        await traffic_system.broadcast_state()
                        return {"status": "success", "message": "Optimization applied successfully"}
                    else:
                        raise HTTPException(status_code=400, detail="Failed to apply optimization")
                else:
                    # Log rejection
                    traffic_system.audit_logger.log_recommendation(recommendation, accepted=False)
                    return {"status": "success", "message": "Optimization rejected"}
            else:
                raise HTTPException(status_code=404, detail="Recommendation not found")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid recommendation ID")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply optimization: {str(e)}")

@app.post("/update-conditions")
async def update_conditions(conditions: ConditionUpdate):
    """Update operating conditions"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        condition_dict = conditions.dict(exclude_unset=True)
        traffic_system.optimizer.update_conditions(condition_dict)
        return {"status": "success", "message": "Conditions updated", "conditions": condition_dict}
    except Exception as e:
        logger.error(f"Failed to update conditions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update conditions: {str(e)}")

@app.get("/ml-predictions")
async def get_ml_predictions():
    """Get ML ETA predictions"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        predictions = traffic_system.get_ml_predictions()
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Failed to get ML predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")

@app.get("/optimization-recommendations")
async def get_optimization_recommendations():
    """Get current optimization recommendations"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        recommendations = traffic_system.get_optimization_recommendations()
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.get("/audit-logs")
async def get_audit_logs(limit: int = 50):
    """Get audit trail logs"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        logs = traffic_system.audit_logger.get_recent_logs(limit)
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit logs: {str(e)}")

@app.get("/kpi-history")
async def get_kpi_history(hours: int = 24):
    """Get KPI history"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        history = traffic_system.audit_logger.get_kpi_history(hours)
        return {"history": history}
    except Exception as e:
        logger.error(f"Failed to get KPI history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get KPI history: {str(e)}")

@app.get("/enhanced-metrics")
async def get_enhanced_metrics():
    """Get enhanced performance metrics"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        recommendations_ratio = 0
        if traffic_system.enhanced_metrics.get('total_recommendations', 0) > 0:
            recommendations_ratio = (
                traffic_system.enhanced_metrics.get('recommendations_accepted', 0) / 
                traffic_system.enhanced_metrics.get('total_recommendations', 1)
            )
        
        return {
            "basic_metrics": traffic_system.metrics,
            "enhanced_metrics": traffic_system.enhanced_metrics,
            "ml_accuracy": traffic_system.enhanced_metrics.get('ml_accuracy', 0),
            "recommendations_ratio": recommendations_ratio
        }
    except Exception as e:
        logger.error(f"Failed to get enhanced metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhanced metrics: {str(e)}")

@app.get("/station-status")
async def get_station_status():
    """Get detailed station status information"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        stations = [s for s in TRACK_SECTIONS if s['type'] == 'station']
        station_status = {}
        
        for station in stations:
            platforms = traffic_system.station_platforms.get(station['id'], {})
            occupied = sum(1 for occupant in platforms.values() if occupant is not None)
            total = len(platforms)
            
            station_status[station['id']] = {
                'name': station['name'],
                'station_code': station['station'],
                'platforms': platforms,
                'occupied_platforms': occupied,
                'total_platforms': total,
                'occupancy_percentage': (occupied / total * 100) if total > 0 else 0,
                'status': 'full' if occupied == total else 'partial' if occupied > 0 else 'free'
            }
        
        return {"station_status": station_status}
    except Exception as e:
        logger.error(f"Failed to get station status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get station status: {str(e)}")

@app.get("/network-overview")
async def get_network_overview():
    """Get comprehensive network overview"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
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
        
        network_utilization = 0
        if (total_blocks + total_platforms) > 0:
            network_utilization = ((occupied_blocks + occupied_platforms) / (total_blocks + total_platforms) * 100)
        
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
                "network_utilization": network_utilization
            },
            "simulation_status": {
                "is_running": traffic_system.is_running,
                "simulation_time": traffic_system.simulation_time,
                "simulation_time_formatted": f"{traffic_system.simulation_time // 60:02d}:{traffic_system.simulation_time % 60:02d}"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get network overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get network overview: {str(e)}")

@app.post("/inject-accident")
async def inject_accident(accident: AccidentInjection):
    """Inject accident scenario"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        success = traffic_system.inject_accident(accident.section_id, accident.duration)
        if not success:
            raise HTTPException(status_code=400, detail="Accident injection failed - section may already be disrupted")
            
        await traffic_system.broadcast_state()
        return {"status": "success", "message": f"Accident injected in {accident.section_id} for {accident.duration} minutes"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject accident: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to inject accident: {str(e)}")

@app.post("/inject-breakdown")
async def inject_breakdown(breakdown: BreakdownInjection):
    """Inject train breakdown scenario"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        success = traffic_system.inject_breakdown(breakdown.train_id, breakdown.duration)
        if not success:
            raise HTTPException(status_code=404, detail="Train not found or already disrupted")
            
        await traffic_system.broadcast_state()
        return {"status": "success", "message": f"Breakdown injected for train {breakdown.train_id} for {breakdown.duration} minutes"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject breakdown: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to inject breakdown: {str(e)}")

@app.post("/inject-track-closure")
async def inject_track_closure(closure: TrackClosureInjection):
    """Inject track closure scenario"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        success = traffic_system.inject_track_closure(closure.section_id, closure.duration)
        if not success:
            raise HTTPException(status_code=400, detail="Track closure failed - section may already be closed")
            
        await traffic_system.broadcast_state()
        return {"status": "success", "message": f"Track closed: {closure.section_id} for {closure.duration} minutes"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject track closure: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to inject track closure: {str(e)}")

@app.post("/simulate-scenario")
async def simulate_scenario(scenario: ScenarioSimulation):
    """Simulate complex scenarios"""
    if not traffic_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        # Mock scenario simulation - can be enhanced with real scenario logic
        mock_kpis = {
            "avgDelay": 5.2,
            "throughput": 8.5,
            "efficiency_improvement": 12.3,
            "scenario_type": scenario.conditions.get("type", "unknown")
        }
        return {"scenario_results": mock_kpis, "status": "success"}
    except Exception as e:
        logger.error(f"Failed to simulate scenario: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate scenario: {str(e)}")

@app.get("/system-status")
async def get_system_status():
    """Get overall system health status"""
    try:
        status = {
            "system_initialized": traffic_system is not None,
            "simulation_running": traffic_system.is_running if traffic_system else False,
            "ml_model_ready": traffic_system.ml_predictor.is_trained if traffic_system else False,
            "websocket_connections": len(traffic_system.websocket_connections) if traffic_system else 0,
            "total_trains": len(traffic_system.trains) if traffic_system else 0,
            "system_time": traffic_system.simulation_time if traffic_system else 0
        }
        return {"system_status": status}
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_ready": traffic_system is not None,
        "version": "6.0.1"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Intelligent Railway Control System",
        "version": "6.0.1",
        "status": "operational",
        "features": [
            "Real-time train tracking",
            "ML-powered ETA predictions",
            "Dynamic schedule optimization",
            "Scenario simulation",
            "Comprehensive audit trail"
        ]
    }

if __name__ == "__main__": 
    logger.info("Starting Enhanced Railway Control System...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )