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
    <head>
        <title>GreenOps-X</title>
        <style>
    body {
        font-family: 'Segoe UI', monospace;
        background: #0f172a;
        color: #e2e8f0;
        margin: 0;
        padding: 20px;
    }
    h1 { color: #22c55e; }

    .container { max-width: 800px; margin: auto; }

    .card {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }

    button {
        background: #22c55e;
        color: black;
        border: none;
        padding: 10px 15px;
        margin: 5px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: bold;
    }

    input {
        padding: 10px;
        width: 60%;
        border-radius: 6px;
        border: none;
        margin-right: 10px;
    }

    pre {
        background: black;
        padding: 15px;
        border-radius: 8px;
        color: #22c55e;
    }

    .metrics {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }

    .metric {
        background: #020617;
        padding: 10px;
        border-radius: 6px;
        flex: 1;
        margin: 5px;
        text-align: center;
    }
</style>
    </head>

    <body>
        <div class="container">
            <h1>🔥 GreenOps-X Environment</h1>
            <p>AI-powered data center thermal optimization</p>

            <p style="color:#94a3b8">
            System dynamically balances thermal load and power efficiency using AI policies.
            </p>
            
            <div class="card">
                <button onclick="reset()">Reset Environment</button>

                <br><br>

                <input id="action" placeholder="e.g. migrate_jobs(0,2)" />
                <button onclick="step()">Execute Action</button>
            </div>

            <div class="card">
                <h3>📊 System State</h3>

                <div class="metrics">
                    <div class="metric">
                        <b>Rack Temps</b>
                        <div id="temps">-</div>
                    </div>
                    <div class="metric">
                        <b>CPU Load</b>
                        <div id="loads">-</div>
                    </div>
                    <div class="metric">
                        <b>Power Cost</b>
                        <div id="power">-</div>
                    </div>
                </div>

                <div class="metrics">
                    <div class="metric">
                        <b>Reward</b>
                        <div id="reward">-</div>
                    </div>
                    <div class="metric">
                        <b>Status</b>
                        <div id="status">-</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🧾 Action Log</h3>
                <pre id="log"></pre>
            </div>
            
            <div class="card">
                <h3>📦 Raw Output</h3>
                <pre id="output"></pre>
            </div>
        </div>

        <script>
            let history = [];
let tempHistory = [];

async function reset() {
    let res = await fetch('/reset');
    let data = await res.json();
    history = [];
    tempHistory = [];
    render(data);
}

async function step() {
    let action = document.getElementById('action').value;

    let res = await fetch('/step?action=' + action);
    let data = await res.json();

    history.push(action);
    document.getElementById('log').innerText = history.join("\n");

    render(data);
}

function getColor(temp) {
    if (temp > 85) return "#ef4444";
    if (temp > 75) return "#f59e0b";
    return "#22c55e";
}

function render(data) {
    tempHistory.push(data.observation.rack_temp);

    // Raw JSON output
    document.getElementById('output').innerText =
        JSON.stringify(data, null, 2);

    // Temps with color
    document.getElementById('temps').innerHTML =
        (data.observation?.rack_temp || [])
        .map(t => `<span style="color:${getColor(t)}">${t.toFixed(1)}°C</span>`)
        .join(" | ");

    // Load formatted
    document.getElementById('loads').innerText =
        (data.observation?.cpu_load || []).map(x => x.toFixed(2)).join(" | ");

    document.getElementById('power').innerText =
        data.observation?.power_cost?.toFixed(2) || "-";

    document.getElementById('reward').innerText =
        data.reward?.toFixed(3) ?? "-";

    // Status
    if (data.observation?.failed_fan) {
        document.getElementById('status').innerHTML =
            "<span style='color:red'>⚠ FAN FAILURE</span>";
    } else {
        document.getElementById('status').innerText =
            data.done ? "Finished" : "Running";
    }
}
        </script>
    </body>
    </html>
    """
    
if __name__ == "__main__":
    main()