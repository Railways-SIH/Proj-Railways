# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

from .core.system import EnhancedTrafficControlSystem, add_default_trains
from .api.apiendpoints import register_api_endpoints

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Intelligent Railway Control Backend", version="6.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://proj-railways.onrender.com", "https://railways-project.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

# Initialize the enhanced system globally
traffic_system = EnhancedTrafficControlSystem()
# NOTE: add_default_trains is called inside system.reset_simulation, 
# which is implicitly called when the system is instantiated.
# We call it again here just to ensure the initial load happens before startup_event tries to access it.
add_default_trains(traffic_system) 

# Register all API routes
register_api_endpoints(app, traffic_system)

# --- Simulation Loop ---

async def simulation_loop():
    while True:
        if traffic_system.is_running: 
            await traffic_system.update_simulation()
        await asyncio.sleep(1.5)

@app.on_event("startup")
async def startup_event(): 
    # The default trains were already added during instantiation/global scope.
    # Start the simulation background task.
    asyncio.create_task(simulation_loop())

# NOTE: The uvicorn running code is moved to uvicorn_start.py