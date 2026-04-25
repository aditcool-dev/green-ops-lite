from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from env.environment import GreenOpsEnv
from env.grader import grade
from audit_trail import AuditTrail
from sla import SLAMonitor
import uvicorn
import os

app = FastAPI(title="GreenOps-X")

# ── Shared environment ──────────────────────────────────────
env      = GreenOpsEnv()
_history = []
_audit   = AuditTrail()
_sla     = SLAMonitor()

# ── Routes ──────────────────────────────────────────────────

@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.api_route("/grade", methods=["GET", "POST"])
async def get_grade():
    return {"score": round(grade(env), 4)}


@app.api_route("/reset", methods=["GET", "POST"])
async def reset(request: Request):
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

    # ── NEW: reset audit + SLA for fresh episode ──
    _audit.clear()
    _sla.start_episode(task)

    return {"observation": obs.dict(), "task": task}


@app.api_route("/step", methods=["GET", "POST"])
async def step(request: Request):
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

    # snapshot BEFORE step for audit
    obs_before = env._get_obs()

    result = env.step(action)

    # ── NEW: record in audit trail ──
    _audit.record(
        step       = env.step_count,
        action     = action,
        obs_before = obs_before,
        obs_after  = result.observation,
        reward     = result.reward,
        done       = result.done,
    )

    # ── NEW: record in SLA monitor ──
    _sla.record_step(
        step       = env.step_count,
        temps      = list(result.observation.rack_temp),
        loads      = list(result.observation.cpu_load),
        power_cost = result.observation.power_cost,
        reward     = result.reward,
    )

    # live alerts for UI
    alerts = _sla.live_alerts(
        temps      = list(result.observation.rack_temp),
        power_cost = result.observation.power_cost,
    )

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

    response = {
        "observation": result.observation.dict(),
        "reward":      result.reward,
        "done":        result.done,
        "alerts":      alerts,
        "audit_hash":  _audit.latest(1)[0]["hash_short"] if len(_audit) > 0 else None,
    }

    # if episode done, attach SLA report
    if result.done:
        final_score = grade(env)
        sla_report  = _sla.evaluate(final_score)
        response["sla_report"] = {
            "tier":        sla_report.tier,
            "tier_label":  sla_report.tier_label,
            "tier_color":  sla_report.tier_color,
            "tier_icon":   sla_report.tier_icon,
            "grade_score": sla_report.grade_score,
            "peak_temp":   sla_report.peak_temp,
            "passed":      sla_report.passed_checks,
            "failed":      sla_report.failed_checks,
            "improvement": sla_report.improvement_vs_baseline,
        }

    return response


@app.get("/state")
async def get_state():
    return env.state()


@app.get("/history")
async def get_history():
    return {"history": _history}


# ── NEW endpoints ────────────────────────────────────────────

@app.get("/audit")
async def get_audit():
    """Full cryptographic ledger export."""
    return _audit.export()


@app.get("/audit/verify")
async def verify_audit():
    """Verify chain integrity — recomputes every hash."""
    return _audit.verify()


@app.get("/audit/latest")
async def get_audit_latest():
    """Last 5 audit entries (lightweight, for UI ticker)."""
    return {"entries": _audit.latest(5), "total": len(_audit)}


@app.get("/sla")
async def get_sla():
    """Current SLA report (or in-progress summary)."""
    return _sla.current_report()


@app.get("/sla/comparison")
async def get_sla_comparison():
    """Static pre/post fine-tuning SLA comparison table."""
    return SLAMonitor.comparison_table()


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
body::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.07) 2px,rgba(0,0,0,.07) 4px);
}
.shell { max-width: 1200px; margin: 0 auto; padding: 20px 16px; display: grid; gap: 16px; }

/* header */
.header { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--border); padding-bottom:12px; }
.header h1 { font-family:var(--font-ui); font-size:1.3rem; font-weight:700; color:var(--accent); letter-spacing:3px; text-shadow:0 0 18px rgba(0,212,255,.5); }
.header h1 span { color:var(--green); }
.tag { font-size:.65rem; padding:2px 8px; border-radius:2px; background:rgba(0,212,255,.08); color:var(--accent); border:1px solid rgba(0,212,255,.25); letter-spacing:1px; }

/* cards */
.card { background:var(--card); border:1px solid var(--border); border-radius:6px; padding:16px; position:relative; overflow:hidden; }
.card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,var(--accent),transparent); opacity:.4; }
.card-title { font-family:var(--font-ui); font-size:.6rem; letter-spacing:2px; color:var(--dim); margin-bottom:12px; display:flex; align-items:center; gap:6px; }
.card-title .dot { width:6px; height:6px; border-radius:50%; background:var(--accent); box-shadow:0 0 6px var(--accent); animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

.main-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media(max-width:740px){ .main-grid{ grid-template-columns:1fr; } }

/* controls */
.controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
select { background:#051020; color:var(--text); border:1px solid var(--border); border-radius:4px; padding:8px 12px; font-family:var(--font-mono); font-size:.85rem; cursor:pointer; }
select:focus { outline:1px solid var(--accent); }
.btn { padding:8px 18px; border:none; border-radius:4px; font-family:var(--font-ui); font-size:.65rem; letter-spacing:1.5px; cursor:pointer; transition:all .15s; }
.btn-primary { background:var(--accent); color:#000; font-weight:700; }
.btn-primary:hover { box-shadow:0 0 14px rgba(0,212,255,.6); }
.btn-danger { background:transparent; color:var(--red); border:1px solid var(--red); }
.btn-danger:hover { background:rgba(255,59,59,.1); }
.action-row { display:flex; gap:8px; flex:1; }
.action-input { flex:1; background:#051020; color:var(--green); border:1px solid var(--border); border-radius:4px; padding:8px 12px; font-family:var(--font-mono); font-size:.9rem; }
.action-input:focus { outline:1px solid var(--green); }
.action-input::placeholder { color:var(--dim); }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.chip { padding:4px 10px; border-radius:3px; font-size:.7rem; cursor:pointer; border:1px solid transparent; transition:all .12s; user-select:none; }
.chip-cool  { background:rgba(0,212,255,.1); color:var(--accent); border-color:rgba(0,212,255,.2); }
.chip-load  { background:rgba(245,158,11,.1); color:var(--amber); border-color:rgba(245,158,11,.2); }
.chip-mig   { background:rgba(168,85,247,.1); color:#a855f7; border-color:rgba(168,85,247,.2); }
.chip:hover { filter:brightness(1.3); }

/* racks */
.racks { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.rack { background:#040e1c; border:1px solid var(--border); border-radius:5px; padding:12px; display:flex; flex-direction:column; align-items:center; gap:8px; position:relative; }
.rack-id { font-family:var(--font-ui); font-size:.6rem; letter-spacing:2px; color:var(--dim); }
.gauge-track { width:28px; height:80px; background:#020b14; border-radius:3px; border:1px solid #0d3a5c; display:flex; align-items:flex-end; overflow:hidden; }
.gauge-fill { width:100%; border-radius:2px; transition:height .4s, background .4s; }
.rack-temp { font-family:var(--font-ui); font-size:1rem; font-weight:700; transition:color .3s; }
.rack-load { font-size:.7rem; color:var(--dim); }
.fan-badge { position:absolute; top:6px; right:6px; font-size:.55rem; background:rgba(255,59,59,.15); color:var(--red); border:1px solid rgba(255,59,59,.3); padding:2px 5px; border-radius:2px; display:none; }

/* KPIs */
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:14px; }
.kpi { background:#040e1c; border:1px solid #0d3a5c; border-radius:4px; padding:10px 12px; }
.kpi-label { font-size:.6rem; color:var(--dim); letter-spacing:1px; margin-bottom:4px; }
.kpi-value { font-family:var(--font-ui); font-size:1rem; font-weight:700; }

/* chart */
.chart-container { position:relative; height:160px; }

/* log */
.log { height:120px; overflow-y:auto; font-size:.75rem; display:flex; flex-direction:column; gap:4px; }
.log::-webkit-scrollbar { width:4px; }
.log::-webkit-scrollbar-thumb { background:#0d3a5c; border-radius:2px; }
.log-entry { display:flex; gap:10px; padding:3px 0; border-bottom:1px solid rgba(13,58,92,.4); align-items:baseline; }
.log-step   { color:var(--dim); width:28px; flex-shrink:0; }
.log-action { color:var(--accent); flex:1; }
.log-reward { width:58px; text-align:right; flex-shrink:0; font-weight:700; }

.status-bar { display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border); font-size:.72rem; }
.status-pill { padding:2px 8px; border-radius:2px; font-size:.6rem; letter-spacing:1px; font-family:var(--font-ui); }
.status-ok   { background:rgba(0,255,136,.1); color:var(--green); border:1px solid rgba(0,255,136,.2); }
.status-warn { background:rgba(245,158,11,.1); color:var(--amber); border:1px solid rgba(245,158,11,.2); }
.status-crit { background:rgba(255,59,59,.1);  color:var(--red);   border:1px solid rgba(255,59,59,.2); }
.status-fan  { background:rgba(255,59,59,.15); color:var(--red);   border:1px solid rgba(255,59,59,.35); animation:pulse 1s infinite; }

/* ── ALERT PANEL ── */
#alert-panel { display:none; }
.alert-item { display:flex; align-items:center; gap:8px; padding:6px 10px; border-radius:4px; margin-bottom:6px; font-size:.75rem; }
.alert-dot  { width:8px; height:8px; border-radius:50%; flex-shrink:0; animation:pulse 1s infinite; }

/* ── AUDIT TICKER ── */
.audit-ticker { font-size:.68rem; display:flex; flex-direction:column; gap:5px; max-height:90px; overflow:hidden; }
.audit-entry { display:flex; gap:8px; align-items:center; padding:4px 6px; background:rgba(0,212,255,.04); border-radius:3px; border-left:2px solid rgba(0,212,255,.3); }
.audit-seq   { color:var(--dim); width:24px; flex-shrink:0; }
.audit-act   { color:var(--green); flex:1; }
.audit-hash  { color:var(--dim); font-size:.62rem; }
.verify-btn  { font-size:.6rem; padding:3px 8px; background:transparent; color:var(--accent); border:1px solid rgba(0,212,255,.3); border-radius:3px; cursor:pointer; font-family:var(--font-mono); margin-top:6px; }
.verify-btn:hover { background:rgba(0,212,255,.08); }
.verify-result { font-size:.68rem; margin-top:6px; padding:5px 8px; border-radius:3px; }
.verify-ok   { background:rgba(0,255,136,.08); color:var(--green); border:1px solid rgba(0,255,136,.2); }
.verify-fail { background:rgba(255,59,59,.08); color:var(--red);   border:1px solid rgba(255,59,59,.2); }

/* ── SLA PANEL ── */
.sla-tier { display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:5px; margin-bottom:10px; border:1px solid; }
.sla-icon { font-size:1.4rem; }
.sla-name { font-family:var(--font-ui); font-size:.75rem; letter-spacing:2px; font-weight:700; }
.sla-desc { font-size:.68rem; color:var(--dim); margin-top:2px; }
.sla-checks { font-size:.68rem; display:flex; flex-direction:column; gap:3px; }
.sla-check-ok   { color:var(--green); }
.sla-check-fail { color:var(--red); }
.sla-comparison { width:100%; border-collapse:collapse; font-size:.7rem; margin-top:8px; }
.sla-comparison th { color:var(--dim); text-align:left; padding:4px 8px; border-bottom:1px solid var(--border); font-weight:400; letter-spacing:1px; font-size:.62rem; }
.sla-comparison td { padding:5px 8px; border-bottom:1px solid rgba(13,58,92,.4); }
</style>
</head>
<body>
<div class="shell">

  <!-- header -->
  <div class="header">
    <h1>GREEN<span>OPS</span>-X</h1>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <span class="tag" id="task-tag">TASK: EASY</span>
      <span class="tag" id="audit-count-tag">AUDIT: 0 entries</span>
    </div>
  </div>

  <!-- controls -->
  <div class="card">
    <div class="card-title"><span class="dot"></span>MISSION CONTROL</div>
    <div class="controls">
      <select id="task-select">
        <option value="easy">🟢 EASY</option>
        <option value="medium">🟡 MEDIUM</option>
        <option value="hard">🔴 HARD</option>
      </select>
      <button class="btn btn-primary" onclick="doReset()">RESET</button>
      <div class="action-row">
        <input class="action-input" id="action-input" placeholder="increase_cooling(0)">
        <button class="btn btn-primary" onclick="doStep()">EXECUTE</button>
      </div>
    </div>
    <div class="chips">
      <span class="chip chip-cool" onclick="setAction('increase_cooling(0)')">cool(0)</span>
      <span class="chip chip-cool" onclick="setAction('increase_cooling(1)')">cool(1)</span>
      <span class="chip chip-cool" onclick="setAction('increase_cooling(2)')">cool(2)</span>
      <span class="chip chip-load" onclick="setAction('decrease_load(0)')">load↓(0)</span>
      <span class="chip chip-load" onclick="setAction('decrease_load(1)')">load↓(1)</span>
      <span class="chip chip-load" onclick="setAction('decrease_load(2)')">load↓(2)</span>
      <span class="chip chip-mig"  onclick="setAction('migrate_jobs(0,2)')">migrate(0→2)</span>
      <span class="chip chip-mig"  onclick="setAction('migrate_jobs(1,2)')">migrate(1→2)</span>
      <span class="chip chip-mig"  onclick="setAction('migrate_jobs(2,0)')">migrate(2→0)</span>
    </div>
  </div>

  <!-- alert panel (hidden when no alerts) -->
  <div class="card" id="alert-panel">
    <div class="card-title"><span class="dot" style="background:var(--red);box-shadow:0 0 6px var(--red);"></span>LIVE ALERTS</div>
    <div id="alert-list"></div>
  </div>

  <!-- main grid -->
  <div class="main-grid">

    <!-- left: racks + chart -->
    <div class="card">
      <div class="card-title"><span class="dot"></span>RACK STATUS</div>
      <div class="racks">
        <div class="rack">
          <div class="fan-badge" id="fan-0">FAN FAIL</div>
          <div class="rack-id">RACK 0</div>
          <div class="gauge-track"><div class="gauge-fill" id="gauge-0" style="height:40%;background:var(--green)"></div></div>
          <div class="rack-temp" id="temp-0">—</div>
          <div class="rack-load" id="load-0">load —</div>
        </div>
        <div class="rack">
          <div class="rack-id">RACK 1</div>
          <div class="gauge-track"><div class="gauge-fill" id="gauge-1" style="height:50%;background:var(--green)"></div></div>
          <div class="rack-temp" id="temp-1">—</div>
          <div class="rack-load" id="load-1">load —</div>
        </div>
        <div class="rack">
          <div class="rack-id">RACK 2</div>
          <div class="gauge-track"><div class="gauge-fill" id="gauge-2" style="height:30%;background:var(--green)"></div></div>
          <div class="rack-temp" id="temp-2">—</div>
          <div class="rack-load" id="load-2">load —</div>
        </div>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-label">POWER COST</div><div class="kpi-value" id="kpi-power" style="color:var(--green)">—</div></div>
        <div class="kpi"><div class="kpi-label">STEP REWARD</div><div class="kpi-value" id="kpi-reward" style="color:var(--green)">—</div></div>
        <div class="kpi"><div class="kpi-label">STEP</div><div class="kpi-value" id="kpi-step" style="color:var(--accent)">0</div></div>
        <div class="kpi"><div class="kpi-label">GRADE SCORE</div><div class="kpi-value" id="kpi-grade" style="color:var(--green)">—</div></div>
      </div>
      <div style="margin-top:12px">
        <div class="card-title"><span class="dot"></span>TEMPERATURE HISTORY</div>
        <div class="chart-container"><canvas id="temp-chart"></canvas></div>
      </div>
    </div>

    <!-- right: log + audit + SLA -->
    <div style="display:flex;flex-direction:column;gap:16px;">

      <!-- action log -->
      <div class="card">
        <div class="card-title"><span class="dot"></span>ACTION LOG</div>
        <div class="log" id="log"></div>
        <div class="status-bar">
          <span id="status-text">AWAITING RESET</span>
          <span class="status-pill status-ok" id="status-pill">IDLE</span>
        </div>
      </div>

      <!-- audit trail -->
      <div class="card">
        <div class="card-title">
          <span class="dot" style="background:#a855f7;box-shadow:0 0 6px #a855f7;"></span>
          CRYPTOGRAPHIC AUDIT TRAIL
        </div>
        <div class="audit-ticker" id="audit-ticker">
          <div style="color:var(--dim);font-size:.7rem;">No actions recorded yet. Each action is SHA-256 hashed and chain-linked.</div>
        </div>
        <button class="verify-btn" onclick="verifyChain()">⬡ VERIFY CHAIN INTEGRITY</button>
        <div id="verify-result" style="display:none;"></div>
      </div>

      <!-- SLA panel -->
      <div class="card">
        <div class="card-title">
          <span class="dot" style="background:#FFD700;box-shadow:0 0 6px #FFD700;"></span>
          SLA COMPLIANCE
        </div>
        <div id="sla-live">
          <div style="color:var(--dim);font-size:.7rem;">SLA report generated at episode end.</div>
          <div style="margin-top:10px;">
            <div style="font-size:.65rem;color:var(--dim);margin-bottom:6px;letter-spacing:1px;">PRE vs POST FINE-TUNING</div>
            <table class="sla-comparison" id="sla-comparison-table">
              <thead><tr><th>TASK</th><th>PRE SCORE</th><th>PRE TIER</th><th>POST SCORE</th><th>POST TIER</th><th>Δ</th></tr></thead>
              <tbody id="sla-comparison-body"></tbody>
            </table>
          </div>
        </div>
        <div id="sla-report" style="display:none;margin-top:10px;"></div>
      </div>

    </div>
  </div>

</div>

<script>
// ── State ──────────────────────────────────────────────────
let stepCount = 0;
let tempChart = null;
const LABELS = [], D0 = [], D1 = [], D2 = [];

// ── Chart ─────────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById('temp-chart').getContext('2d');
  if (tempChart) tempChart.destroy();
  LABELS.length = D0.length = D1.length = D2.length = 0;
  tempChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: LABELS,
      datasets: [
        { label:'R0', data:D0, borderColor:'#00d4ff', backgroundColor:'rgba(0,212,255,.05)', tension:.4, pointRadius:2, borderWidth:1.5 },
        { label:'R1', data:D1, borderColor:'#00ff88', backgroundColor:'rgba(0,255,136,.05)', tension:.4, pointRadius:2, borderWidth:1.5 },
        { label:'R2', data:D2, borderColor:'#a855f7', backgroundColor:'rgba(168,85,247,.05)', tension:.4, pointRadius:2, borderWidth:1.5 },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false, animation:{ duration:200 },
      plugins:{ legend:{ labels:{ color:'#4a7a9b', font:{ family:'Share Tech Mono', size:10 } } } },
      scales:{
        x:{ ticks:{ color:'#4a7a9b', font:{ size:9 } }, grid:{ color:'#061526' } },
        y:{ min:50, max:110, ticks:{ color:'#4a7a9b', font:{ size:9 } }, grid:{ color:'rgba(13,58,92,.5)' } }
      }
    }
  });
}

// ── Gauge / rack helpers ───────────────────────────────────
function tempColor(t) {
  if (t >= 90) return '#ff3b3b';
  if (t >= 75) return '#f59e0b';
  return '#00ff88';
}
function updateRack(i, temp, load, fanFailed) {
  const pct = Math.max(0, Math.min(100, ((temp - 50) / 60) * 100));
  const col = tempColor(temp);
  document.getElementById(`temp-${i}`).textContent = temp.toFixed(1) + '°C';
  document.getElementById(`temp-${i}`).style.color  = col;
  document.getElementById(`load-${i}`).textContent  = 'load ' + load.toFixed(2);
  const fill = document.getElementById(`gauge-${i}`);
  fill.style.height     = pct + '%';
  fill.style.background = col;
  if (i === 0) document.getElementById('fan-0').style.display = fanFailed ? '' : 'none';
}

// ── KPIs ──────────────────────────────────────────────────
function updateKPIs(obs, reward) {
  const pc = obs.power_cost || 0;
  document.getElementById('kpi-power').textContent = pc.toFixed(2);
  document.getElementById('kpi-power').style.color = pc > 1.5 ? 'var(--red)' : pc > 1.0 ? 'var(--amber)' : 'var(--green)';
  const rEl = document.getElementById('kpi-reward');
  if (reward !== null && reward !== undefined) {
    rEl.textContent = reward.toFixed(3);
    rEl.style.color = reward < 0 ? 'var(--red)' : reward < 0.2 ? 'var(--amber)' : 'var(--green)';
  }
  document.getElementById('kpi-step').textContent = stepCount;
}

function setStatus(text, pill, cls) {
  document.getElementById('status-text').textContent = text;
  const p = document.getElementById('status-pill');
  p.textContent = pill;
  p.className   = 'status-pill ' + cls;
}

// ── Grade ─────────────────────────────────────────────────
async function fetchGrade() {
  try {
    const r = await fetch('/grade');
    const d = await r.json();
    const s = d.score;
    const el = document.getElementById('kpi-grade');
    el.textContent = s.toFixed(4);
    el.style.color = s > 0.45 ? 'var(--green)' : s > 0.35 ? 'var(--amber)' : 'var(--red)';
  } catch(e) {}
}

// ── Render ────────────────────────────────────────────────
function render(obs, reward, done) {
  const temps = obs.rack_temp || [0,0,0];
  const loads = obs.cpu_load  || [0,0,0];
  const fan   = obs.failed_fan || false;
  for (let i = 0; i < 3; i++) updateRack(i, temps[i], loads[i], fan);
  updateKPIs(obs, reward);
  LABELS.push(stepCount);
  D0.push(temps[0]); D1.push(temps[1]); D2.push(temps[2]);
  if (tempChart) tempChart.update();
  const taskEl = document.getElementById('task-tag');
  taskEl.textContent = 'TASK: ' + (document.getElementById('task-select').value || 'EASY').toUpperCase();
  const maxT = Math.max(...temps);
  if (fan)          setStatus('⚠ RACK 0 FAN FAILURE — HEAT CASCADE RISK', 'FAN FAIL', 'status-fan');
  else if (done)    setStatus('Episode complete — reset to continue', 'DONE', 'status-ok');
  else if (maxT>90) setStatus('CRITICAL: temp '+maxT.toFixed(1)+'°C — immediate action required', 'CRITICAL', 'status-crit');
  else if (maxT>75) setStatus('WARNING: temp '+maxT.toFixed(1)+'°C rising', 'CAUTION', 'status-warn');
  else              setStatus('System nominal — step '+stepCount, 'NOMINAL', 'status-ok');
}

// ── Log ───────────────────────────────────────────────────
function logAction(action, reward) {
  const log  = document.getElementById('log');
  const rCol = reward < 0 ? 'var(--red)' : reward < 0.2 ? 'var(--amber)' : 'var(--green)';
  const el   = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span class="log-step">[${String(stepCount).padStart(2,'0')}]</span>
    <span class="log-action">${action}</span>
    <span class="log-reward" style="color:${rCol}">${reward >= 0 ? '+' : ''}${reward.toFixed(3)}</span>`;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

// ── NEW: Alerts ───────────────────────────────────────────
function renderAlerts(alerts) {
  const panel = document.getElementById('alert-panel');
  const list  = document.getElementById('alert-list');
  if (!alerts || alerts.length === 0) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';
  list.innerHTML = alerts.map(a => `
    <div class="alert-item" style="background:${a.color}18;border:1px solid ${a.color}44;">
      <div class="alert-dot" style="background:${a.color};box-shadow:0 0 6px ${a.color};"></div>
      <span style="color:${a.color};font-weight:700;width:72px;flex-shrink:0;">${a.level}</span>
      <span style="color:var(--text);">${a.message}</span>
    </div>`).join('');
}

// ── NEW: Audit ticker ─────────────────────────────────────
function renderAuditTicker(entries, total) {
  const ticker = document.getElementById('audit-ticker');
  document.getElementById('audit-count-tag').textContent = `AUDIT: ${total} entries`;
  if (!entries || entries.length === 0) return;
  ticker.innerHTML = entries.map(e => `
    <div class="audit-entry">
      <span class="audit-seq">#${e.seq}</span>
      <span class="audit-act">${e.action}</span>
      <span style="color:${e.reward >= 0 ? 'var(--green)':'var(--red)'};">${e.reward >= 0?'+':''}${e.reward.toFixed(3)}</span>
      <span class="audit-hash">${e.hash_short}</span>
    </div>`).join('');
}

async function verifyChain() {
  const btn = document.querySelector('.verify-btn');
  btn.textContent = '⬡ VERIFYING…';
  try {
    const r = await fetch('/audit/verify');
    const d = await r.json();
    const el = document.getElementById('verify-result');
    el.style.display = 'block';
    el.className = 'verify-result ' + (d.valid ? 'verify-ok' : 'verify-fail');
    el.textContent = d.valid
      ? `✓ CHAIN INTACT — ${d.entries} entries verified. Tip: ${d.chain_tip || ''}`
      : `✗ TAMPERING DETECTED at seq=${d.first_tampered_seq} — ${d.message}`;
  } catch(e) {
    console.error(e);
  }
  btn.textContent = '⬡ VERIFY CHAIN INTEGRITY';
}

// ── NEW: SLA ──────────────────────────────────────────────
function renderSLAReport(report) {
  if (!report) return;
  const el = document.getElementById('sla-report');
  el.style.display = 'block';
  const imp = report.improvement;
  const impHtml = imp ? `
    <div style="margin-top:8px;padding:6px 8px;background:rgba(0,255,136,.05);border-radius:3px;border:1px solid rgba(0,255,136,.15);font-size:.68rem;">
      <span style="color:var(--dim);">vs baseline (${imp.task}):</span>
      <span style="color:var(--green);margin-left:8px;">${imp.baseline_tier} → ${imp.current_tier}</span>
      <span style="color:var(--dim);margin-left:8px;">${imp.baseline_score} → ${imp.current_score}</span>
      <span style="color:var(--green);margin-left:8px;">${imp.delta > 0 ? '+' : ''}${imp.delta} (${imp.delta > 0 ? '+' : ''}${imp.delta_pct}%)</span>
    </div>` : '';

  const passedHtml = (report.passed || []).map(c =>
    `<div class="sla-check-ok">✓ ${c}</div>`).join('');
  const failedHtml = (report.failed || []).map(c =>
    `<div class="sla-check-fail">✗ ${c}</div>`).join('');

  el.innerHTML = `
    <div class="sla-tier" style="background:${report.tier_color}14;border-color:${report.tier_color}44;color:${report.tier_color};">
      <span class="sla-icon">${report.tier_icon}</span>
      <div>
        <div class="sla-name">${report.tier_label.toUpperCase()} SLA</div>
        <div class="sla-desc" style="color:${report.tier_color}99;">${report.tier_description || ''}</div>
      </div>
    </div>
    <div class="sla-checks">${passedHtml}${failedHtml}</div>
    ${impHtml}`;
}

async function loadSLAComparison() {
  try {
    const r = await fetch('/sla/comparison');
    const d = await r.json();
    const body = document.getElementById('sla-comparison-body');
    if (!body || !d.comparison) return;
    body.innerHTML = d.comparison.map(row => `
      <tr>
        <td style="color:var(--text);">${row.task}</td>
        <td style="color:var(--dim);">${row.pre_score}</td>
        <td style="color:${row.pre_tier_color};font-weight:700;">${row.pre_tier}</td>
        <td style="color:var(--dim);">${row.post_score}</td>
        <td style="color:${row.post_tier_color};font-weight:700;">${row.post_tier}</td>
        <td style="color:var(--green);">+${row.delta}</td>
      </tr>`).join('');
  } catch(e) {}
}

// ── API ───────────────────────────────────────────────────
async function doReset() {
  const task = document.getElementById('task-select').value;
  try {
    const res = await fetch('/reset', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ task })
    });
    if (!res.ok) throw new Error('Reset failed: '+res.status);
    const data = await res.json();
    stepCount = 0;
    document.getElementById('log').innerHTML = '';
    document.getElementById('kpi-grade').textContent = '—';
    document.getElementById('alert-panel').style.display = 'none';
    document.getElementById('sla-report').style.display = 'none';
    document.getElementById('verify-result').style.display = 'none';
    document.getElementById('audit-ticker').innerHTML =
      '<div style="color:var(--dim);font-size:.7rem;">Audit cleared. New episode started.</div>';
    document.getElementById('audit-count-tag').textContent = 'AUDIT: 0 entries';
    initChart();
    render(data.observation, null, false);
    setStatus('Environment reset · task='+task, 'READY', 'status-ok');
  } catch(e) {
    setStatus('ERROR: '+e.message, 'ERROR', 'status-crit');
  }
}

async function doStep() {
  const action = document.getElementById('action-input').value.trim();
  if (!action) { document.getElementById('action-input').focus(); return; }
  try {
    const res = await fetch('/step', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ action })
    });
    if (!res.ok) throw new Error('Step failed: '+res.status);
    const data = await res.json();
    stepCount++;
    render(data.observation, data.reward, data.done);
    logAction(action, data.reward);

    // alerts
    renderAlerts(data.alerts || []);

    // audit ticker update
    try {
      const ar = await fetch('/audit/latest');
      const ad = await ar.json();
      renderAuditTicker(ad.entries, ad.total);
    } catch(e) {}

    if (stepCount % 2 === 0) fetchGrade();

    if (data.done) {
      fetchGrade();
      setStatus('Episode finished — click RESET', 'DONE', 'status-ok');
      if (data.sla_report) renderSLAReport(data.sla_report);
    }
  } catch(e) {
    setStatus('ERROR: '+e.message, 'ERROR', 'status-crit');
  }
}

function setAction(a) {
  document.getElementById('action-input').value = a;
  document.getElementById('action-input').focus();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.activeElement.id === 'action-input') doStep();
});

// ── Boot ──────────────────────────────────────────────────
initChart();
loadSLAComparison();
doReset();
</script>
</body>
</html>""")


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()