from fastapi import FastAPI
from env.environment import GreenOpsEnv

app = FastAPI()

env = GreenOpsEnv()

@app.get("/")
def home():
    return {
        "status": "running",
        "endpoints": ["/reset", "/step?action=..."]
    }

@app.get("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.dict()}

@app.get("/step")
def step(action: str):
    result = env.step(action)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done
    }