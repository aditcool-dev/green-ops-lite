from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from env.environment import GreenOpsEnv
from env.grader import grade
import uvicorn
import os

app = FastAPI(title="GreenOps-X")

# ── Shared environment ──────────────────────────────────────
env      = GreenOpsEnv()
_history = []   # list of {action, reward, temps, loads, step}


# ── Routes ──────────────────────────────────────────────────

@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.api_route("/grade", methods=["GET", "POST"])
async def get_grade():
    return {"score": round(grade(env), 4)}


@app.api_route("/reset", methods=["GET", "POST"])
async def reset(request: Request):
    """
    [BUG-1 FIX] Accept task from POST body OR query param.
    Also clears server-side history so /history is fresh.
    """
    global _history
    task = "easy"

    if request.method == "POST":
        try:
            data = await request.json()
            task = data.get("task", "easy") if isinstance(data, dict) else "easy"
        except Exception:
            task = request.query_params.get("task", "easy")
    else:
        task = request.query_params.get("task", "easy")

    obs      = env.reset(task)
    _history = []
    return {"observation": obs.dict(), "task": task}


@app.api_route("/step", methods=["GET", "POST"])
async def step(request: Request):
    """
    [BUG-3 FIX] Action now read from POST JSON body.
    GET fallback still works for curl testing.
    """
    global _history

    action = None
    if request.method == "POST":
        try:
            data   = await request.json()
            action = data.get("action") if isinstance(data, dict) else None
        except Exception:
            action = None
    else:
        action = request.query_params.get("action")

    action = action or "increase_cooling(1)"
    result = env.step(action)

    entry = {
        "step":   env.step_count,
        "action": action,
        "reward": result.reward,
        "temps":  list(result.observation.rack_temp),
        "loads":  list(result.observation.cpu_load),
        "power":  result.observation.power_cost,
        "done":   result.done,
    }
    _history.append(entry)

    return {
        "observation": result.observation.dict(),
        "reward":      result.reward,
        "done":        result.done,
    }


@app.get("/state")
async def get_state():
    return env.state()


@app.get("/history")
async def get_history():
    """Endpoint for JS chart — returns full step history."""
    return {"history": _history}


# ── UI ──────────────────────────────────────────────────────

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GreenOps-X · Thermal Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ── Reset & base ───────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #020b14;
  --card:     #061526;
  --border:   #0d3a5c;
  --accent:   #00d4ff;
  --green:    #00ff88;
  --amber:    #f59e0b;
  --red:      #ff3b3b;
  --text:     #c8e6ff;
  --dim:      #4a7a9b;
  --font-mono:'Share Tech Mono', monospace;
  --font-ui:  'Orbitron', monospace;
}

html, body {
  height: 100%; background: var(--bg);
  color: var(--text); font-family: var(--font-mono);
  overflow-x: hidden;
}

/* scanlines */
body::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,.07) 2px, rgba(0,0,0,.07) 4px
  );
}

/* ── Layout ─────────────────────────────────────────────── */
.shell {
  max-width: 1200px; margin: 0 auto; padding: 20px 16px;
  display: grid; gap: 16px;
}

/* header */
.header {
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border); padding-bottom: 12px;
}
.header h1 {
  font-family: var(--font-ui); font-size: 1.3rem; font-weight: 700;
  color: var(--accent); letter-spacing: 3px;
  text-shadow: 0 0 18px rgba(0,212,255,.5);
}
.header h1 span { color: var(--green); }
.tag {
  font-size: .65rem; padding: 2px 8px; border-radius: 2px;
  background: rgba(0,212,255,.08); color: var(--accent);
  border: 1px solid rgba(0,212,255,.25); letter-spacing: 1px;
}

/* cards */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px;
  position: relative;
  overflow: hidden;
}
.card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background: linear-gradient(90deg,transparent,var(--accent),transparent);
  opacity:.4;
}
.card-title {
  font-family: var(--font-ui); font-size:.6rem; letter-spacing:2px;
  color: var(--dim); margin-bottom: 12px;
  display: flex; align-items: center; gap: 6px;
}
.card-title .dot {
  width:6px; height:6px; border-radius:50%; background:var(--accent);
  box-shadow: 0 0 6px var(--accent); animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* 2-col main grid */
.main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media(max-width:740px){ .main-grid{ grid-template-columns:1fr; } }

/* ── Controls ───────────────────────────────────────────── */
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }

select {
  background: #051020; color: var(--text);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 8px 12px; font-family: var(--font-mono); font-size:.85rem;
  cursor:pointer;
}
select:focus { outline: 1px solid var(--accent); }

.btn {
  padding: 8px 18px; border: none; border-radius: 4px;
  font-family: var(--font-ui); font-size:.65rem; letter-spacing:1.5px;
  cursor: pointer; transition: all .15s;
}
.btn-primary {
  background: var(--accent); color: #000; font-weight:700;
}
.btn-primary:hover { box-shadow: 0 0 14px rgba(0,212,255,.6); }
.btn-danger {
  background: transparent; color: var(--red);
  border: 1px solid var(--red);
}
.btn-danger:hover { background: rgba(255,59,59,.1); }

.action-row { display:flex; gap:8px; flex:1; }
.action-input {
  flex:1; background:#051020; color: var(--green);
  border: 1px solid var(--border); border-radius:4px;
  padding: 8px 12px; font-family:var(--font-mono); font-size:.9rem;
}
.action-input:focus { outline:1px solid var(--green); }
.action-input::placeholder { color: var(--dim); }

/* quick action chips */
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.chip {
  padding:4px 10px; border-radius:3px; font-size:.7rem; cursor:pointer;
  border: 1px solid; transition: all .12s; font-family:var(--font-mono);
}
.chip-cool  { border-color:var(--accent); color:var(--accent); }
.chip-cool:hover  { background:rgba(0,212,255,.12); }
.chip-mig   { border-color:var(--green);  color:var(--green);  }
.chip-mig:hover   { background:rgba(0,255,136,.12); }
.chip-load  { border-color:var(--amber);  color:var(--amber);  }
.chip-load:hover  { background:rgba(245,158,11,.12); }

/* ── Rack gauges ─────────────────────────────────────────── */
.racks { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }

.rack {
  background:#030e1a; border:1px solid var(--border); border-radius:5px;
  padding:10px; text-align:center;
}
.rack-name {
  font-family:var(--font-ui); font-size:.55rem; letter-spacing:2px;
  color:var(--dim); margin-bottom:6px;
}
.rack-temp {
  font-family:var(--font-ui); font-size:1.5rem; font-weight:700;
  transition: color .3s;
}
.rack-sub { font-size:.7rem; color:var(--dim); margin-top:2px; }

/* vertical bar gauge */
.gauge-wrap { margin:8px auto; width:28px; height:80px;
  background:#010810; border:1px solid var(--border); border-radius:3px;
  display:flex; align-items:flex-end; overflow:hidden; }
.gauge-fill {
  width:100%; border-radius:2px 2px 0 0;
  transition: height .4s ease, background .4s;
}

/* fan indicator */
.fan-badge {
  display:none; margin-top:6px; font-size:.65rem; padding:2px 6px;
  background:rgba(255,59,59,.15); color:var(--red);
  border:1px solid rgba(255,59,59,.3); border-radius:3px;
}

/* ── KPIs ───────────────────────────────────────────────── */
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
@media(max-width:500px){ .kpi-row{ grid-template-columns:1fr 1fr; } }

.kpi {
  background:#030e1a; border:1px solid var(--border); border-radius:5px;
  padding:10px 12px;
}
.kpi-label { font-size:.6rem; color:var(--dim); letter-spacing:1px; }
.kpi-value {
  font-family:var(--font-ui); font-size:1.2rem; font-weight:700;
  margin-top:4px; transition:color .3s;
}

/* ── Chart ───────────────────────────────────────────────── */
.chart-container { position:relative; height:180px; }

/* ── Log ─────────────────────────────────────────────────── */
.log {
  height:160px; overflow-y:auto; font-size:.75rem; line-height:1.8;
  color: var(--dim);
}
.log-entry { display:flex; gap:8px; }
.log-step  { color:var(--dim); min-width:28px; }
.log-action{ color:var(--green); }
.log-reward{ min-width:60px; }

/* ── Status bar ─────────────────────────────────────────── */
.status-bar {
  display:flex; justify-content:space-between; align-items:center;
  font-size:.65rem; color:var(--dim); padding:6px 0 0;
  border-top:1px solid var(--border);
}
.status-pill {
  padding:2px 8px; border-radius:2px; font-family:var(--font-ui);
  font-size:.55rem; letter-spacing:1px;
}
.status-ok   { background:rgba(0,255,136,.1); color:var(--green); border:1px solid rgba(0,255,136,.2); }
.status-warn { background:rgba(245,158,11,.1); color:var(--amber); border:1px solid rgba(245,158,11,.2); }
.status-crit { background:rgba(255,59,59,.1);  color:var(--red);   border:1px solid rgba(255,59,59,.2); }
.status-fan  { background:rgba(255,59,59,.2);  color:var(--red);   border:1px solid var(--red);
               animation:pulse 1s infinite; }

/* scrollbar */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }
</style>
</head>
<body>
<div class="shell">

  <!-- Header -->
  <div class="header">
    <h1>GREEN<span>OPS</span>-X</h1>
    <div style="display:flex;gap:8px;align-items:center">
      <span class="tag">THERMAL CONTROL</span>
      <span class="tag" id="task-tag">TASK: EASY</span>
    </div>
  </div>

  <!-- Controls -->
  <div class="card">
    <div class="card-title"><span class="dot"></span>MISSION CONTROL</div>
    <div class="controls">
      <select id="task-select">
        <option value="easy">EASY</option>
        <option value="medium">MEDIUM</option>
        <option value="hard">HARD — FAN FAILURE</option>
      </select>
      <button class="btn btn-primary" onclick="doReset()">⟳ RESET ENV</button>
      <div class="action-row">
        <input class="action-input" id="action-input"
               placeholder="increase_cooling(0)" />
        <button class="btn btn-primary" onclick="doStep()">EXECUTE ▶</button>
      </div>
    </div>
    <div class="chips">
      <span class="chip chip-cool" onclick="setAction('increase_cooling(0)')">COOL R0</span>
      <span class="chip chip-cool" onclick="setAction('increase_cooling(1)')">COOL R1</span>
      <span class="chip chip-cool" onclick="setAction('increase_cooling(2)')">COOL R2</span>
      <span class="chip chip-mig"  onclick="setAction('migrate_jobs(0,2)')">MIG 0→2</span>
      <span class="chip chip-mig"  onclick="setAction('migrate_jobs(1,2)')">MIG 1→2</span>
      <span class="chip chip-mig"  onclick="setAction('migrate_jobs(2,0)')">MIG 2→0</span>
      <span class="chip chip-load" onclick="setAction('decrease_load(0)')">SHED R0</span>
      <span class="chip chip-load" onclick="setAction('decrease_load(1)')">SHED R1</span>
      <span class="chip chip-load" onclick="setAction('decrease_load(2)')">SHED R2</span>
    </div>
  </div>

  <!-- Rack gauges -->
  <div class="main-grid">
    <div class="card">
      <div class="card-title"><span class="dot"></span>RACK THERMALS</div>
      <div class="racks">
        <div class="rack" id="rack-0">
          <div class="rack-name">RACK 0</div>
          <div class="gauge-wrap"><div class="gauge-fill" id="gauge-0"></div></div>
          <div class="rack-temp" id="temp-0">—</div>
          <div class="rack-sub" id="load-0">load —</div>
          <div class="fan-badge" id="fan-0">⚠ FAN FAIL</div>
        </div>
        <div class="rack" id="rack-1">
          <div class="rack-name">RACK 1</div>
          <div class="gauge-wrap"><div class="gauge-fill" id="gauge-1"></div></div>
          <div class="rack-temp" id="temp-1">—</div>
          <div class="rack-sub" id="load-1">load —</div>
          <div class="fan-badge" id="fan-1" style="display:none"></div>
        </div>
        <div class="rack" id="rack-2">
          <div class="rack-name">RACK 2</div>
          <div class="gauge-wrap"><div class="gauge-fill" id="gauge-2"></div></div>
          <div class="rack-temp" id="temp-2">—</div>
          <div class="rack-sub" id="load-2">load —</div>
          <div class="fan-badge" id="fan-2" style="display:none"></div>
        </div>
      </div>
    </div>

    <!-- KPIs + score -->
    <div class="card">
      <div class="card-title"><span class="dot"></span>METRICS</div>
      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-label">POWER COST</div>
          <div class="kpi-value" id="kpi-power">—</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">LAST REWARD</div>
          <div class="kpi-value" id="kpi-reward">—</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">STEP</div>
          <div class="kpi-value" id="kpi-step">0</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">GRADE SCORE</div>
          <div class="kpi-value" id="kpi-grade" style="color:var(--green)">—</div>
        </div>
      </div>
      <div style="margin-top:12px">
        <div class="card-title"><span class="dot"></span>TEMPERATURE HISTORY</div>
        <div class="chart-container">
          <canvas id="temp-chart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Action log -->
  <div class="card">
    <div class="card-title"><span class="dot"></span>ACTION LOG</div>
    <div class="log" id="log"></div>
    <div class="status-bar">
      <span id="status-text">AWAITING RESET</span>
      <span class="status-pill status-ok" id="status-pill">IDLE</span>
    </div>
  </div>

</div>

<script>
// ── State ──────────────────────────────────────────────────
let stepCount  = 0;
let tempChart  = null;
const LABELS   = [];
const D0 = [], D1 = [], D2 = [];

// ── Chart init ─────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById('temp-chart').getContext('2d');
  if (tempChart) tempChart.destroy();
  LABELS.length = D0.length = D1.length = D2.length = 0;

  tempChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: LABELS,
      datasets: [
        { label:'R0', data:D0, borderColor:'#00d4ff', backgroundColor:'rgba(0,212,255,.05)',
          tension:.4, pointRadius:2, borderWidth:1.5 },
        { label:'R1', data:D1, borderColor:'#00ff88', backgroundColor:'rgba(0,255,136,.05)',
          tension:.4, pointRadius:2, borderWidth:1.5 },
        { label:'R2', data:D2, borderColor:'#a855f7', backgroundColor:'rgba(168,85,247,.05)',
          tension:.4, pointRadius:2, borderWidth:1.5 },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:{ duration:200 },
      plugins:{ legend:{ labels:{ color:'#4a7a9b', font:{ family:'Share Tech Mono', size:10 } } } },
      scales:{
        x:{ ticks:{ color:'#4a7a9b', font:{ size:9 } }, grid:{ color:'#061526' } },
        y:{ min:50, max:110, ticks:{ color:'#4a7a9b', font:{ size:9 } },
            grid:{ color:'rgba(13,58,92,.5)' } }
      }
    }
  });
}

// ── Gauge helpers ──────────────────────────────────────────
function tempColor(t) {
  if (t >= 90) return '#ff3b3b';
  if (t >= 75) return '#f59e0b';
  return '#00ff88';
}
function updateRack(i, temp, load, fanFailed) {
  const pct  = Math.max(0, Math.min(100, ((temp - 50) / 60) * 100));
  const col  = tempColor(temp);
  document.getElementById(`temp-${i}`).textContent = temp.toFixed(1) + '°C';
  document.getElementById(`temp-${i}`).style.color  = col;
  document.getElementById(`load-${i}`).textContent  = 'load ' + load.toFixed(2);
  const fill = document.getElementById(`gauge-${i}`);
  fill.style.height     = pct + '%';
  fill.style.background = col;
  // Fan badge only on rack 0 when fan fails
  if (i === 0) {
    document.getElementById('fan-0').style.display = fanFailed ? '' : 'none';
  }
}

// ── KPI helpers ────────────────────────────────────────────
function updateKPIs(obs, reward) {
  const pc = obs.power_cost || 0;
  document.getElementById('kpi-power').textContent  = pc.toFixed(2);
  document.getElementById('kpi-power').style.color  = pc > 1.5 ? 'var(--red)' : pc > 1.0 ? 'var(--amber)' : 'var(--green)';
  const rEl = document.getElementById('kpi-reward');
  if (reward !== null && reward !== undefined) {
    rEl.textContent  = reward.toFixed(3);
    rEl.style.color  = reward < 0 ? 'var(--red)' : reward < 0.2 ? 'var(--amber)' : 'var(--green)';
  }
  document.getElementById('kpi-step').textContent = stepCount;
}

function setStatus(text, pill, cls) {
  document.getElementById('status-text').textContent = text;
  const p = document.getElementById('status-pill');
  p.textContent  = pill;
  p.className    = 'status-pill ' + cls;
}

// ── Grade fetch (after each step) ─────────────────────────
async function fetchGrade() {
  try {
    const r = await fetch('/grade');
    const d = await r.json();
    const s = d.score;
    const el = document.getElementById('kpi-grade');
    el.textContent = s.toFixed(4);
    el.style.color = s > 0.45 ? 'var(--green)' : s > 0.35 ? 'var(--amber)' : 'var(--red)';
  } catch(e) { /* ignore */ }
}

// ── Render ─────────────────────────────────────────────────
function render(obs, reward, done) {
  const temps  = obs.rack_temp  || [0,0,0];
  const loads  = obs.cpu_load   || [0,0,0];
  const fan    = obs.failed_fan || false;

  for (let i = 0; i < 3; i++) updateRack(i, temps[i], loads[i], fan);
  updateKPIs(obs, reward);

  // chart
  LABELS.push(stepCount);
  D0.push(temps[0]); D1.push(temps[1]); D2.push(temps[2]);
  if (tempChart) tempChart.update();

  // task tag
  const taskEl = document.getElementById('task-tag');
  taskEl.textContent = 'TASK: ' + (obs.task_name || document.getElementById('task-select').value || 'EASY').toUpperCase();

  // status
  const maxT = Math.max(...temps);
  if (fan)          setStatus('⚠ RACK 0 FAN FAILURE — HEAT CASCADE RISK', 'FAN FAIL', 'status-fan');
  else if (done)    setStatus('Episode complete — reset to continue', 'DONE', 'status-ok');
  else if (maxT>90) setStatus('CRITICAL: temp ' + maxT.toFixed(1)+'°C — immediate action required', 'CRITICAL', 'status-crit');
  else if (maxT>75) setStatus('WARNING: temp ' + maxT.toFixed(1)+'°C rising', 'CAUTION', 'status-warn');
  else              setStatus('System nominal — step ' + stepCount, 'NOMINAL', 'status-ok');
}

// ── Log ────────────────────────────────────────────────────
function logAction(action, reward) {
  const log  = document.getElementById('log');
  const type = action.startsWith('increase') ? 'chip-cool' :
               action.startsWith('migrate')  ? 'chip-mig'  : 'chip-load';
  const rCol = reward < 0 ? 'var(--red)' : reward < 0.2 ? 'var(--amber)' : 'var(--green)';
  const el   = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span class="log-step">[${String(stepCount).padStart(2,'0')}]</span>
    <span class="log-action">${action}</span>
    <span class="log-reward" style="color:${rCol}">${reward >= 0 ? '+' : ''}${reward.toFixed(3)}</span>`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

// ── API calls ──────────────────────────────────────────────
async function doReset() {
  const task = document.getElementById('task-select').value;
  try {
    const res = await fetch('/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task })
    });
    if (!res.ok) throw new Error('Reset failed: ' + res.status);
    const data = await res.json();
    stepCount = 0;
    document.getElementById('log').innerHTML = '';
    document.getElementById('kpi-grade').textContent = '—';
    initChart();
    render(data.observation, null, false);
    setStatus('Environment reset · task=' + task, 'READY', 'status-ok');
  } catch(e) {
    setStatus('ERROR: ' + e.message, 'ERROR', 'status-crit');
    console.error(e);
  }
}

async function doStep() {
  const action = document.getElementById('action-input').value.trim();
  if (!action) { document.getElementById('action-input').focus(); return; }
  try {
    const res = await fetch('/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
    if (!res.ok) throw new Error('Step failed: ' + res.status);
    const data = await res.json();
    stepCount++;
    render(data.observation, data.reward, data.done);
    logAction(action, data.reward);
    if (stepCount % 2 === 0) fetchGrade();  // grade every 2 steps (cheaper)
    if (data.done) {
      fetchGrade();
      setStatus('Episode finished — click RESET to start a new episode', 'DONE', 'status-ok');
    }
  } catch(e) {
    setStatus('ERROR: ' + e.message, 'ERROR', 'status-crit');
    console.error(e);
  }
}

function setAction(a) {
  document.getElementById('action-input').value = a;
  document.getElementById('action-input').focus();
}

// keyboard shortcut: Enter = execute
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.activeElement.id === 'action-input') doStep();
});

// ── Boot ───────────────────────────────────────────────────
initChart();
doReset();   // load initial state on page open
</script>
</body>
</html>""")


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()