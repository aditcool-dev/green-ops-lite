from fastapi import FastAPI
from fastapi import Request
from env.environment import GreenOpsEnv
import uvicorn
import os

app = FastAPI()

env = GreenOpsEnv()

@app.get("/")
def home():
    return {
        "status": "running",
        "endpoints": ["/reset", "/step?action=..."]
    }

@app.post("/reset")
async def reset(request: Request):
    data = await request.json() if request.method == "POST" else {}
    task = data.get("task", "easy")

    obs = env.reset(task)
    return {"observation": obs.dict()}

@app.post("/step")
async def step(request: Request):
    data = await request.json()
    action = data.get("action")

    result = env.step(action)

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done
    }
    

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()