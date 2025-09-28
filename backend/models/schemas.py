# backend/models/schemas.py

from pydantic import BaseModel
from typing import List, Optional

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