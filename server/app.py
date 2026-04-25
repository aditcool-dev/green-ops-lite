from fastapi import FastAPI
from fastapi import Request
from env.environment import GreenOpsEnv
import uvicorn
import os
from env.grader import grade
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI()

env = GreenOpsEnv()

@app.get("/")
def root():
    return RedirectResponse(url="/ui")

@app.api_route("/grade", methods=["GET", "POST"])
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
            task = "easy"
    else:
        task = request.query_params.get("task", "easy")

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

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <body style="font-family: monospace; background: #111; color: #0f0;">
        <h2>GreenOps-X Environment</h2>

        <button onclick="reset()">Reset</button>
        <br><br>

        <input id="action" placeholder="e.g. increase_cooling(0)" />
        <button onclick="step()">Step</button>

        <pre id="output"></pre>

        <script>
            async function reset() {
                let res = await fetch('/reset');
                let data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            }

            async function step() {
                let action = document.getElementById('action').value;
                let res = await fetch('/step?action=' + action);
                let data = await res.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 2);
            }
        </script>
    </body>
    </html>
    """
    
if __name__ == "__main__":
    main()