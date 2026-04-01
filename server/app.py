from fastapi import FastAPI
from fastapi import Request
from env.environment import GreenOpsEnv
import uvicorn
import os
from env.grader import grade

app = FastAPI()

env = GreenOpsEnv()

@app.get("/")
def home():
    return {
        "status": "running",
        "endpoints": ["/reset", "/step?action=..."]
    }

@app.get("/grade")
async def get_grade():
    score = grade(env)
    return {"score": score}

@app.api_route("/reset", methods=["GET", "POST"])
async def reset(request: Request):
    task = "easy"

    if request.method == "POST":
        try:
            data = await request.json()
            if isinstance(data, dict):
                task = data.get("task", "easy")
        except:
            # 🔥 VERY IMPORTANT: avoid crash
            task = "easy"

    obs = env.reset(task)
    return {"observation": obs.dict()}

@app.api_route("/step", methods=["GET", "POST"])
async def step(request: Request):
    action = None

    if request.method == "POST":
        try:
            data = await request.json()
            if isinstance(data, dict):
                action = data.get("action")
        except:
            action = None
    else:
        action = request.query_params.get("action")

    # 🔥 fallback (VERY IMPORTANT)
    if not action:
        action = "increase_cooling(1)"

    result = env.step(action)

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done
    }

@app.get("/state")
async def get_state():
    return env.state()

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()