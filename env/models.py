from pydantic import BaseModel
from typing import List, Dict

class Observation(BaseModel):
    rack_temp: List[float]
    cpu_load: List[float]
    power_cost: float
    failed_fan: bool
    step_count: int


class Action(BaseModel):
    action_type: str
    source_rack: int | None = None
    target_rack: int | None = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict