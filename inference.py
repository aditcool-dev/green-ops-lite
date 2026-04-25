import os
import re
import json
import math
import time
from collections import deque
from openai import OpenAI
from env.environment import GreenOpsEnv
from env.grader import grade
import numpy as np
import random, time
import os as _os
random.seed(int.from_bytes(_os.urandom(8), "big"))
from unsloth import FastLanguageModel
import torch

# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "unsloth/mistral-7b-instruct"

# ============================================================
# ACTOR MODEL
# ============================================================
_actor_model, _actor_tokenizer = FastLanguageModel.from_pretrained(
    model_name="Adit555/greenops-actor-lora",
    max_seq_length=1024,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(_actor_model)

# ============================================================
# OVERSEER MODEL
# ============================================================
_overseer_model, _overseer_tokenizer = FastLanguageModel.from_pretrained(
    model_name="Adit555/greenops-overseer-lora",
    max_seq_length=1024,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(_overseer_model)

# [FEAT-14] Different step budgets per task
MAX_STEPS = {"easy": 10, "medium": 10, "hard": 10}
DEBUG        = os.getenv("DEBUG", "false").lower() == "true"
N_RACKS   = 3
NUM_RACKS = 3
_last_space_fix = None

# ============================================================
# LOGGING
# ============================================================

def debug_log(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)


def log_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             role_rewards: dict | None = None) -> None:
    base = (f"[STEP] step={step} action={action} "
            f"reward={reward:.4f} done={str(done).lower()}")
    if role_rewards:
        rr = " ".join(f"{k}={v:.3f}" for k, v in role_rewards.items())
        base += f" role_rewards={rr}"
    print(base, flush=True)


def log_end(score: float) -> None:
    print(f"[END] score={score:.4f}", flush=True)


# ============================================================
# ACTION UTILITIES
# ============================================================

_ACTION_PATTERN = re.compile(
    r"(increase_cooling\(\d\)|decrease_load\(\d\)|migrate_jobs\(\d,\s*\d\))"
)


def extract_action(text: str | None) -> str | None:
    """Parse the first valid action from an arbitrary string."""
    if not text:
        return None
    m = _ACTION_PATTERN.search(text)
    return m.group(0) if m else None


def get_rack(action: str | None) -> int:
    """Return the primary rack index from an action string, or -1 on failure."""
    if not action:
        return -1
    try:
        return int(re.findall(r"\d+", action)[0])
    except (IndexError, ValueError):
        return -1


def get_source_rack(action: str | None) -> int:
    return get_rack(action)


def get_dest_rack(action: str | None) -> int:
    """For migrate_jobs(src, dst) return dst; otherwise -1."""
    if not action or "migrate_jobs" not in action:
        return -1
    try:
        nums = re.findall(r"\d+", action)
        return int(nums[1])
    except (IndexError, ValueError):
        return -1


def conflict_score(a1: str | None, a2: str | None) -> float:
    """
    Structural conflict detection.
    Identical actions (consensus) → 0.0.
    Contradictory migrations → 1.0.
    Same-source different-target → 0.4.
    """
    if not a1 or not a2:
        return 0.0
    if a1.replace(" ", "") == a2.replace(" ", ""):
        return 0.0
    r1, r2 = get_rack(a1), get_rack(a2)
    if r1 == -1 or r2 == -1:
        return 0.0
    t1 = a1.split("(")[0]
    t2 = a2.split("(")[0]
    if t1 == "migrate_jobs" and t2 == "migrate_jobs":
        src1, dst1 = r1, get_dest_rack(a1)
        src2, dst2 = r2, get_dest_rack(a2)
        if src1 == dst2 and src2 == dst1:
            return 1.0
        if src1 == src2 and dst1 != dst2:
            return 0.4
    return 0.0


def should_use_overseer(obs, confidence: float, c_score: float = 0.0) -> bool:
    """Gate the second LLM pass — only fire when genuinely needed."""
    max_temp = max(obs.rack_temp)
    fan      = getattr(obs, "failed_fan", False)
    if c_score > 0:        return True
    if fan:                return True
    if max_temp >= 83.0:   return True
    if confidence < 0.75:  return True
    return False


# ============================================================
# [FEAT-1] THERMAL STATE MACHINE WITH HYSTERESIS
# ============================================================

class ThermalStateMachine:
    """
    Hysteresis-based thermal zone tracker.
    Zones: SAFE → CONTROL → HIGH → EMERGENCY
    Hysteresis band: 5C between entry and exit of each zone.
    """

    SAFE      = "safe"
    CONTROL   = "control"
    HIGH      = "high"
    EMERGENCY = "emergency"

    _TRANSITIONS = {
        SAFE:      dict(up=75.0, up_zone=CONTROL),
        CONTROL:   dict(up=85.0, up_zone=HIGH,      dn=70.0, dn_zone=SAFE),
        HIGH:      dict(up=92.0, up_zone=EMERGENCY,  dn=80.0, dn_zone=CONTROL),
        EMERGENCY: dict(                              dn=87.0, dn_zone=HIGH),
    }

    def __init__(self):
        self.state = self.SAFE

    def update(self, max_temp: float) -> str:
        t = self._TRANSITIONS.get(self.state, {})
        if "up" in t and max_temp >= t["up"]:
            self.state = t["up_zone"]
        elif "dn" in t and max_temp < t["dn"]:
            self.state = t["dn_zone"]
        return self.state

    def reset(self):
        self.state = self.SAFE


# ============================================================
# [FEAT-2] RATE-OF-CHANGE DETECTOR
# ============================================================

class RateOfChangeDetector:
    """Tracks per-rack temperature rate of change to trigger early intervention."""

    def __init__(self, window: int = 2, max_history: int = 6):
        self.window  = window
        self.history: deque = deque(maxlen=max_history)

    def record(self, temps: list) -> None:
        self.history.append(list(temps))

    def roc(self, rack_idx: int) -> float:
        """C per step over the last `window` steps for a given rack."""
        if len(self.history) < self.window + 1:
            return 0.0
        old = self.history[-(self.window + 1)][rack_idx]
        new = self.history[-1][rack_idx]
        return (new - old) / self.window

    def max_roc(self) -> tuple[int, float]:
        """Return (rack_idx, roc) for the rack with the fastest rise."""
        if len(self.history) < 2:
            return 0, 0.0
        rocs = [(i, self.roc(i)) for i in range(len(self.history[-1]))]
        return max(rocs, key=lambda x: x[1])

    def reset(self):
        self.history.clear()


# ============================================================
# [FEAT-3] CASCADE PREDICTOR — PHYSICS LOOKAHEAD
# ============================================================

class CascadePredictor:
    """
    Physics model matching GreenOpsEnv dynamics exactly.
    Calibrated from env/environment.py:
      heat_factor = 2.5      (cpu_load[i] * 2.5)
      runaway     = min(5.0, exp((t-75)*0.05)-1) when t > 75
      failed_fan: rack0 +3.0, rack1 +1.5, rack2 +0.0
    """

    def __init__(self, heat_factor: float = 2.5,
                 rack0_bleed: float = 3.0,
                 rack1_bleed: float = 1.5,
                 runaway_threshold: float = 75.0):
        self.heat_factor       = heat_factor
        self.rack0_bleed       = rack0_bleed
        self.rack1_bleed       = rack1_bleed
        self.runaway_threshold = runaway_threshold

    def predict(self, temps: list, loads: list, failed_fan: bool = False) -> list:
        """Return predicted temperatures after one step — mirrors env physics exactly."""
        next_temps = []
        for i, (t, l) in enumerate(zip(temps, loads)):
            delta = l * self.heat_factor
            if t > self.runaway_threshold:
                delta += min(5.0, math.exp((t - self.runaway_threshold) * 0.05) - 1)
            if failed_fan:
                if i == 0:   delta += self.rack0_bleed
                elif i == 1: delta += self.rack1_bleed
            next_temps.append(round(t + delta, 2))
        return next_temps

    def will_cascade(self, temps: list, loads: list, failed_fan: bool,
                     danger_threshold: float = 85.0) -> tuple[bool, int]:
        predicted = self.predict(temps, loads, failed_fan)
        for i, pt in enumerate(predicted):
            if pt >= danger_threshold:
                return True, i
        return False, -1


# ============================================================
# [FEAT-4] ADAPTIVE THRESHOLDS — EMA-TUNED PER EPISODE
# ============================================================

class AdaptiveThresholds:
    """Adjusts zone thresholds dynamically based on observed episode heat rate."""

    def __init__(self, alpha: float = 0.3):
        self.alpha    = alpha
        self.avg_temp = 55.0
        self.avg_roc  = 0.0

    def update(self, max_temp: float, roc: float) -> None:
        self.avg_temp = self.alpha * max_temp + (1 - self.alpha) * self.avg_temp
        self.avg_roc  = self.alpha * abs(roc) + (1 - self.alpha) * self.avg_roc

    @property
    def control_entry(self) -> float:
        base = 75.0
        if self.avg_temp > 70:  base -= 4.0
        if self.avg_roc  > 3.0: base -= 3.0
        return max(58.0, base)

    @property
    def high_entry(self) -> float:
        return min(88.0, self.control_entry + 11.0)

    @property
    def migrate_imbalance_threshold(self) -> float:
        base = 6.0
        if self.avg_temp > 74:  base = 5.0
        if self.avg_roc  > 2.0: base = 4.0
        return base

    def reset(self):
        self.avg_temp = 55.0
        self.avg_roc  = 0.0


# ============================================================
# [FEAT-5] ACTION MEMORY — HISTORY, TREND, THRASH DETECTION
# ============================================================

class ActionMemory:
    """Thrash detection, reward trend, action history for LLM context."""

    def __init__(self, maxlen: int = 15):
        self.actions: deque = deque(maxlen=maxlen)
        self.rewards: deque = deque(maxlen=maxlen)

    def record_action(self, action: str) -> None:
        self.actions.append(action)

    def record_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def last_action(self) -> str | None:
        return self.actions[-1] if self.actions else None

    def is_thrashing(self, candidate: str, window: int = 4) -> bool:
        """True when the candidate action has dominated the recent window (>60%)."""
        if len(self.actions) < window:
            return False
        recent = list(self.actions)[-window:]
        return sum(1 for a in recent if a == candidate) / window > 0.60

    def reward_trend(self, window: int = 3) -> float:
        """Slope over last `window` rewards. Negative = declining."""
        r = list(self.rewards)
        if len(r) < window + 1:
            return 0.0
        return r[-1] - r[-(window + 1)]

    def recent_rewards(self, n: int = 3) -> list:
        r = list(self.rewards)
        return r[-n:] if len(r) >= n else r

    def recent_actions(self, n: int = 3) -> list:
        a = list(self.actions)
        return a[-n:] if len(a) >= n else a

    def reset(self):
        self.actions.clear()
        self.rewards.clear()


# ============================================================
# [FEAT-6] RACK HEALTH MONITOR
# ============================================================

class RackHealthMonitor:
    """
    Composite per-rack health score (0.0 = critical, 1.0 = healthy).
    [FIX-D] Hard temp veto: racks above 82C are excluded as migration targets.
             At 82C+ a rack is already in thermal trouble — sending more load
             accelerates its cascade rather than distributing heat.
    """

    TEMP_CEILING = 95.0
    LOAD_CEILING = 1.0

    def score(self, temp: float, load: float, roc: float) -> float:
        temp_margin   = max(0.0, (self.TEMP_CEILING - temp) / self.TEMP_CEILING)
        load_margin   = max(0.0, 1.0 - load / self.LOAD_CEILING)
        trend_penalty = max(0.0, min(0.3, roc / 20.0))
        raw = 0.5 * temp_margin + 0.3 * load_margin - trend_penalty
        return round(max(0.0, min(1.0, raw)), 3)

    def best_target(self, temps: list, loads: list, roc_detector: RateOfChangeDetector,
                    exclude: int = -1, min_capacity: float = 0.15,
                    never_target: set | None = None) -> int:
        """
        Return healthiest rack with headroom for new load.
        never_target: racks that must never receive migrations (e.g. {0} in hard mode).
        """
        scores = []
        for i in range(len(temps)):
            if i == exclude:
                scores.append(-1.0); continue
            if never_target and i in never_target:
                scores.append(-1.0); continue
            if loads[i] > (1.0 - min_capacity):
                scores.append(-1.0); continue     # no capacity
            if temps[i] > 82.0:
                scores.append(-1.0); continue     # [FIX-D] too hot to receive load
            roc = roc_detector.roc(i) if len(roc_detector.history) >= 2 else 0.0
            scores.append(self.score(temps[i], loads[i], roc))
        best = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best] <= 0.0:
            # Fallback: coolest rack excluding forbidden targets
            candidates = [i for i in range(len(temps))
                          if i != exclude and (not never_target or i not in never_target)]
            return (min(candidates, key=lambda i: temps[i])
                    if candidates
                    else max(range(len(temps)), key=lambda i: -temps[i]))
        return best


# ============================================================
# [FEAT-7] UCB-1 BANDIT — LEARNS BEST ACTION IN CONTROL ZONE
# ============================================================

class UCB1Bandit:
    """Multi-armed bandit using UCB-1 for action selection in the control zone."""

    def __init__(self, arms: list):
        self.arms   = arms
        self.counts = {a: 0 for a in arms}
        self.totals = {a: 0.0 for a in arms}
        self.t      = 0

    def select(self) -> str:
        self.t += 1
        for a in self.arms:
            if self.counts[a] == 0:
                return a
        def ucb(a):
            avg  = self.totals[a] / self.counts[a]
            conf = math.sqrt(2 * math.log(self.t) / self.counts[a])
            return avg + conf
        return max(self.arms, key=ucb)

    def update(self, arm: str, reward: float) -> None:
        if arm in self.counts:
            self.counts[arm] += 1
            self.totals[arm] += reward

    def reset(self):
        self.counts = {a: 0 for a in self.arms}
        self.totals = {a: 0.0 for a in self.arms}
        self.t      = 0


# ============================================================
# [FEAT-11] EPISODE PROFILER — STRUCTURED TRAINING DATA LOG
# ============================================================

class EpisodeProfiler:
    """Records structured episode data for replay-buffer / training data collection."""

    def __init__(self):
        self.steps: list = []
        self.task: str   = ""

    def start(self, task: str) -> None:
        self.task  = task
        self.steps = []

    def record_step(self, step: int, obs, action: str, reward: float,
                    p1: dict, p2: dict | None, thermal_state: str,
                    predicted_temps: list | None = None) -> None:
        self.steps.append({
            "step":            step,
            "task":            self.task,
            "temps":           [round(t, 1) for t in obs.rack_temp],
            "loads":           [round(l, 2) for l in obs.cpu_load],
            "failed_fan":      getattr(obs, "failed_fan", False),
            "thermal_state":   thermal_state,
            "p1_thermal":      p1.get("thermal"),
            "p1_load":         p1.get("load"),
            "p1_confidence":   round(p1.get("confidence", 0.0), 3),
            "p2_override_t":   p2.get("override_thermal") if p2 else None,
            "p2_override_l":   p2.get("override_load")    if p2 else None,
            "final_action":    action,
            "reward":          round(reward, 4),
            "predicted_temps": predicted_temps,
        })

    def emit_summary(self, final_score: float) -> None:
        summary = {
            "task":             self.task,
            "final_score":      round(final_score, 4),
            "steps":            len(self.steps),
            "avg_reward":       round(sum(s["reward"] for s in self.steps)
                                      / max(1, len(self.steps)), 4),
            "max_temp_seen":    max(max(s["temps"]) for s in self.steps) if self.steps else 0,
            "overseer_invoked": sum(1 for s in self.steps if s["p2_override_t"] is not None),
            "episode_log":      self.steps,
        }
        print(f"[EPISODE_SUMMARY] {json.dumps(summary)}", flush=True)


# ============================================================
# MODULE-LEVEL STATE (reset per episode)
# ============================================================

_tsm      = ThermalStateMachine()
_roc      = RateOfChangeDetector()
_cascade  = CascadePredictor()
_adaptive = AdaptiveThresholds()
_memory   = ActionMemory()
_health   = RackHealthMonitor()
_profiler = EpisodeProfiler()

_bandit_arms = ["increase_cooling", "migrate_jobs", "decrease_load"]
_bandit      = UCB1Bandit(_bandit_arms)

_overseer_fail_count = 0
_DEGRADED_THRESHOLD  = 3
_last_good_p1: dict | None = None


def _reset_episode_state() -> None:
    global _last_good_p1, _overseer_fail_count
    _tsm.reset()
    _roc.reset()
    _adaptive.reset()
    _memory.reset()
    _bandit.reset()
    _last_good_p1        = None
    _overseer_fail_count = 0


# ============================================================
# LLM UTILITIES
# ============================================================

def _llm_call(messages: list, max_tokens: int, temperature: float,
              retries: int = 2,
              _model=None, _tokenizer=None) -> str | None:

    mdl = _model     or _actor_model
    tok = _tokenizer or _actor_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tok(prompt, return_tensors="pt").to(device)

    for attempt in range(retries + 1):
        try:
            with torch.no_grad():
                outputs = mdl.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            return tok.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

        except Exception as e:
            if attempt == retries:
                debug_log(f"LLM failed after {retries+1}: {e}")
                return None
            time.sleep(0.4 * (2 ** attempt))

    return None


def _decompose_reward(obs, action: str, reward: float) -> dict:
    """[FEAT-12] Per-role reward decomposition matching env/environment.py exactly."""
    temps      = obs.rack_temp
    loads      = obs.cpu_load
    stability  = sum(max(0.0, 1.0 - (t - 60.0) / 20.0) for t in temps) / len(temps)
    efficiency = max(0.0, 1.0 - (obs.power_cost / 2.0))
    throughput = sum(loads) / len(loads)
    return {
        "r_stability":  round(0.4 * stability,  3),
        "r_efficiency": round(0.3 * efficiency, 3),
        "r_throughput": round(0.3 * throughput, 3),
    }


# ============================================================
# PASS 1 — THERMAL + LOAD AGENT
# ============================================================

def pass1_llm(obs, predicted_temps: list | None = None,
              thrash_warning: bool = False) -> dict:
    """
    [FEAT-9]  Episode context (last 3 steps + predicted temps) injected into prompt.
    [FEAT-15] No hardcoded example rack values.
    [FIX-B]   thrash_warning: if last 3 actions identical → inject THRASH WARNING.
    [FIX-1/2] Prompt teaches capacity veto (load>=0.85) and zone veto (HIGH/EMERGENCY).
    [FIX-4]   Prompt teaches 86C predicted threshold for early cooling.
    """
    global _last_good_p1

    temps = [round(t, 1) for t in obs.rack_temp]
    loads = [round(l, 2) for l in obs.cpu_load]
    fan   = getattr(obs, "failed_fan", False)

    recent_steps = []
    acts = _memory.recent_actions(3)
    rwds = _memory.recent_rewards(3)
    for i, (a, r) in enumerate(zip(acts, rwds)):
        recent_steps.append({"step": -len(acts) + i, "action": a, "reward": round(r, 3)})

    trend_label = ("improving" if _memory.reward_trend() > 0.05 else
                   "declining" if _memory.reward_trend() < -0.05 else "stable")

    pred_str = str([round(p, 1) for p in predicted_temps]) if predicted_temps else "N/A"

    # [FIX-B] Thrash warning — inject only when the LLM is stuck repeating itself
    thrash_line = (
        "\n*** THRASH WARNING: last 3 actions were IDENTICAL and rewards are declining. "
        "You MUST choose a DIFFERENT action this step. ***\n"
        if thrash_warning else ""
    )

    prompt = f"""You are a data center controller with two roles: Thermal Agent and Load Agent.

CURRENT STATE
  Temps (C): {temps}
  Loads (0-1): {loads}
  failed_fan:  {fan}
  Max temp:    {max(temps)}C
  Thermal zone: {_tsm.state}
  Predicted temps next step: {pred_str}

RECENT HISTORY (last {len(recent_steps)} steps)
  {json.dumps(recent_steps)}
  Reward trend: {trend_label}
{thrash_line}
ACTIONS AVAILABLE
  increase_cooling(N)    — lowers rack N temp by 5C, costs power (hurts efficiency)
  decrease_load(N)       — cuts load on rack N by 0.2, hurts throughput (last resort)
  migrate_jobs(SRC, DST) — moves ~0.3 load from SRC to DST, preserves throughput

MIGRATION RULES — read carefully before choosing migrate_jobs
  - Migrate only if: temp_imbalance > 6C AND dest_load < 0.85 AND dest_temp <= 74C
  - If failed_fan=True AND rack 0 hottest AND load[0] > 0.15 → ALWAYS migrate away from rack 0
  - NEVER migrate TO a rack with load >= 0.85 (capacity full — jobs are silently lost)
  - NEVER migrate TO a rack with temp > 74C (too hot to absorb more load)
  - NEVER migrate in HIGH or EMERGENCY thermal zone — system is too unstable
  - decrease_load is last resort only — penalises throughput score

TEMPERATURE RULES
  - Predicted temp next step >= 86C → thermal_action MUST be increase_cooling on that rack
  - Any rack currently >= 90C → thermal_action MUST be increase_cooling on that rack
  - Any rack currently >= 85C → prefer increase_cooling over migrate
  - Thermal zone HIGH or EMERGENCY → always prefer increase_cooling
  - Below 68C all racks and balanced loads → migrate is acceptable for efficiency

OUTPUT — valid JSON only, no markdown, no explanation:
{{
  "thermal_action": "<increase_cooling(N) | decrease_load(N) | migrate_jobs(N,M)>",
  "load_action":    "<increase_cooling(N) | decrease_load(N) | migrate_jobs(N,M)>",
  "confidence":     <float 0.0-1.0>,
  "reasoning":      "<one short sentence>"
}}"""

    content = _llm_call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.6,
        _model=_actor_model,
        _tokenizer=_actor_tokenizer
    )

    hottest = obs.rack_temp.index(max(obs.rack_temp))
    coolest = obs.rack_temp.index(min(obs.rack_temp))
    thermal, load, confidence = None, None, 0.5

    if content:
        try:
            data       = json.loads(content)
            thermal    = extract_action(str(data.get("thermal_action", "")))
            load       = extract_action(str(data.get("load_action", "")))
            confidence = float(data.get("confidence", 0.5))
            debug_log(f"PASS1 reasoning: {data.get('reasoning', '')}")
        except Exception as e:
            debug_log(f"PASS1 parse error: {e}")

    if not thermal:
        thermal = f"increase_cooling({hottest})"
        debug_log("PASS1 thermal fallback used")
    if not load:
        target = _health.best_target(obs.rack_temp, obs.cpu_load, _roc, exclude=hottest)
        load   = f"migrate_jobs({hottest},{target})"
        debug_log("PASS1 load fallback used")

    result        = {"thermal": thermal, "load": load, "confidence": confidence}
    _last_good_p1 = result
    debug_log(f"PASS1 → {result}")
    return result


# ============================================================
# PASS 2 — OVERSEER AGENT
# ============================================================

def pass2_llm(obs, p1: dict, c_score: float, predicted_temps: list | None) -> dict | None:
    """
    [BUG-6 FIX]  Separate thermal/load override fields.
    [FEAT-10]    Structured output with reason_code enum.
    [BUG-4 FIX]  Failure counter + degraded-mode logging.
    [FIX-4]      Override threshold in prompt updated to 86C.
    """
    global _overseer_fail_count

    temps    = [round(t, 1) for t in obs.rack_temp]
    loads    = [round(l, 2) for l in obs.cpu_load]
    fan      = getattr(obs, "failed_fan", False)
    pred_str = str([round(p, 1) for p in predicted_temps]) if predicted_temps else "N/A"

    prompt = f"""You are a SKEPTICAL safety overseer reviewing decisions made by two other agents.
You did NOT generate these actions. Treat them as a black box from an external source.

CURRENT STATE
  Temps (C): {temps}
  Loads:      {loads}
  failed_fan: {fan}
  Thermal zone: {_tsm.state}
  Conflict score between actions: {c_score} (1.0 = genuine conflict)
  Predicted temps next step: {pred_str}

ACTIONS UNDER REVIEW
  thermal_action: {p1['thermal']}
  load_action:    {p1['load']}
  agent confidence: {round(p1['confidence'], 2)}

YOUR EVALUATION CHECKLIST
  IMPORTANT: conflict_score=0.0 = NO conflict. Never output "conflict_resolved" when score=0.
  IMPORTANT: When max_temp < 83C, migrate_jobs can be better than cooling (no power cost).
             Do NOT override migrate unless predicted_temp_next >= 86C or dest is full.

  Override ONLY when a specific condition is met:
  1. conflict_score > 0.5                → genuine conflict, resolve it
  2. Any rack temp >= 85C AND action is NOT cooling that rack → override to cool hottest
  3. confidence < 0.65                  → very low confidence, reconsider
  4. predicted_temp_next >= 86C         → preemptive cooling needed on hottest predicted rack
  5. failed_fan=True AND load[0] > 0.15 → migrate away from rack 0
  6. Action is migrate AND destination load >= 0.85 → block migration, cool hottest

  If NONE of the above apply → output all_clear, override_thermal=false.

CRITICAL CONSTRAINTS — NEVER VIOLATE:
  - Only 3 racks: rack 0, rack 1, rack 2. NO rack 3+.
  - Valid: increase_cooling(0..2) | decrease_load(0..2) | migrate_jobs(X,Y) X,Y in {{0,1,2}} X!=Y

OUTPUT — valid JSON only:
{{
  "override_thermal":    <true | false>,
  "override_load":       <true | false>,
  "final_thermal":       "<action using ONLY rack indices 0, 1, or 2>",
  "final_load":          "<action using ONLY rack indices 0, 1, or 2>",
  "confidence":          <float 0.0-1.0>,
  "predicted_temp_next": <float, highest predicted temp next step>,
  "reason_code":         "<cascade_detected | conflict_resolved | temp_critical | load_imbalance | fan_failure_response | migration_blocked | all_clear>"
}}

Note: if override_thermal is false, final_thermal MUST equal the original thermal_action.
      if override_load is false, final_load MUST equal the original load_action."""

    content = _llm_call(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=160,
        temperature=0.45,
        _model=_overseer_model,
        _tokenizer=_overseer_tokenizer
    )

    if not content:
        _overseer_fail_count += 1
        if _overseer_fail_count >= _DEGRADED_THRESHOLD:
            print(f"[WARN] degraded_mode=true overseer_failures={_overseer_fail_count}", flush=True)
        return None

    try:
        data   = json.loads(content)
        result = {
            "override_thermal":    bool(data.get("override_thermal", False)),
            "override_load":       bool(data.get("override_load", False)),
            "final_thermal":       extract_action(str(data.get("final_thermal", ""))),
            "final_load":          extract_action(str(data.get("final_load", ""))),
            "confidence":          float(data.get("confidence", 0.5)),
            "predicted_temp_next": data.get("predicted_temp_next"),
            "reason_code":         data.get("reason_code", ""),
        }
        if result.get("predicted_temp_next"):
            debug_log(f"PASS2 predicted_temp_next={result['predicted_temp_next']:.1f}C  "
                      f"reason={result['reason_code']}")
        _overseer_fail_count = max(0, _overseer_fail_count - 1)
        debug_log(f"PASS2 → override_t={result['override_thermal']}  "
                  f"override_l={result['override_load']}")
        return result
    except Exception as e:
        _overseer_fail_count += 1
        debug_log(f"PASS2 parse error: {e}")
        return None


# ============================================================
# MAIN AGENT — DECISION PIPELINE
# ============================================================

def get_action(obs, task) -> str:
    """
    LLM-FIRST decision pipeline with structured safety stops and post-decision vetoes.

    Architecture:
      1. Compute environment state, physics lookahead, thrash detection
      2. Call pass1 LLM (+ thrash warning if needed)
      3. HARD SAFETY STOPS — override for physical constraints
         P0: hard mode, rack 2 full → shed load
         P1: hard mode, rack 0 hottest → evacuate / reduce secondaries / cool rack 0
         P2: any rack > 92C → emergency cool
      4. LLM PRIMARY PATH (conf >= 0.70)
         P4: predicted >= 86C → [FIX-4] early cascade prevention
         P3: overseer + rack correction [FIX-E]
      5. RULE FALLBACK (P6) — when LLM failed
      6. FINALISE + POST-DECISION VETOES
         [FIX-1] Capacity+temp veto: dest load>=0.85 or temp>80C → cool hottest
         [FIX-2] Zone veto: EMERGENCY or HIGH+hot → block migration, cool hottest
    """
    temps    = list(obs.rack_temp)
    loads    = list(obs.cpu_load)
    n        = len(temps)
    fan      = getattr(obs, "failed_fan", False)
    max_temp = max(temps)
    hottest  = temps.index(max_temp)
    
    # Update sensors / state machines
    _roc.record(temps)
    _tsm.update(max_temp)
    roc_rack, roc_val = _roc.max_roc()
    _adaptive.update(max_temp, roc_val)

    # Best migration targets — in hard mode rack 0 is never a destination
    _never_target   = {0} if fan else None
    candidates = sorted(
        [i for i in range(len(loads)) if i != hottest and i != _never_target],
        key=lambda i: (loads[i], temps[i])
    )

    if not candidates:
        # fallback (should rarely happen)
        coolest = int(np.argmin(loads))
    else:
        if random.random() < 0.7:
            coolest = candidates[0]
        else:
            top_k = candidates[:2] if len(candidates) > 1 else candidates
            coolest = random.choice(top_k)

    overall_coolest = temps.index(min(temps))

    # Physics lookahead
    predicted = _cascade.predict(temps, loads, fan)
    pred_max  = max(predicted)
    pred_rack = predicted.index(pred_max)

    # [FIX-B] Thrash detection — warn LLM if it's stuck in a loop
    _candidate_action = f"migrate_jobs({hottest},{coolest})"
    _thrash_warning   = _memory.is_thrashing(_candidate_action, window=3)

    # ─────────────────────────────────────────────────────────────────
    # STEP 1 — CALL LLM (pass 1)
    # ─────────────────────────────────────────────────────────────────
    try:
        p1 = pass1_llm(obs, predicted_temps=predicted, thrash_warning=_thrash_warning)
    except Exception as e:
        debug_log(f"pass1 exception: {e}")
        p1 = _last_good_p1 or {
            "thermal":    f"increase_cooling({hottest})",
            "load":       f"migrate_jobs({hottest},{coolest})",
            "confidence": 0.0,
        }

    c_score    = conflict_score(p1["thermal"], p1["load"])
    llm_conf   = p1.get("confidence", 0.0)
    llm_action = p1["thermal"]

    if c_score > 0:
        debug_log(f"CONFLICT score={c_score:.1f} t={p1['thermal']} l={p1['load']}")

    final_thermal   = None
    final_load      = None
    decision_source = "unknown"

    # ─────────────────────────────────────────────────────────────────
    # STEP 2 — HARD SAFETY STOPS
    # Physical constraints the LLM cannot reliably model.
    # ─────────────────────────────────────────────────────────────────

    # P0: Hard mode — rack 2 full and hot. Shed load to create migration capacity.
    if fan and loads[2] >= 0.95 and temps[2] > 75.0 and temps[0] < 90.0:
        final_thermal   = "decrease_load(2)"
        final_load      = "decrease_load(2)"
        decision_source = "P0_hard_rack2_relief"
        debug_log(f"P0 hard: rack2 full+hot (load={loads[2]:.2f} temp={temps[2]:.1f}C)")
    
    if final_thermal is None:
        # =====================================
        # 🚀 PRE-SATURATION SPACE CONTROL (CRITICAL FIX)
        # =====================================
        if task == "hard" and fan:

            # If secondary racks are getting full → create space EARLY
            secondary_full = sum(1 for i in range(len(loads)) if i != 0 and loads[i] > 0.90)

            if secondary_full >= 1:
                heavy = max(
                    (i for i in range(len(loads)) if i != 0),
                    key=lambda i: loads[i]
                )

                final_thermal = f"decrease_load({heavy})"
                final_load = None
                decision_source = "PREVENT_SATURATION"

                debug_log(f"[CRITICAL FIX] pre-saturation → decrease_load({heavy})")

                return final_thermal
            
        will_cascade, rack = _cascade.will_cascade(temps, loads, fan)

        if will_cascade:
            if rack == 0 and fan:

                # 🚀 CRITICAL FIX: don't stop evacuation too early
                if loads[0] > 0.10:
                    final_thermal   = f"migrate_jobs(0,{coolest})"
                    final_load      = f"migrate_jobs(0,{coolest})"
                    decision_source = "PREEMPTIVE_EVIAC"
                
                elif temps[0] > 100:
                    # still dangerous → mix strategy
                    final_thermal   = "increase_cooling(0)"
                    final_load      = None
                    decision_source = "CRITICAL_COOLING"

                else:
                    # only AFTER almost empty → cooling makes sense
                    final_thermal   = "increase_cooling(0)"
                    final_load      = None
                    decision_source = "STABILIZE"

                debug_log(f"[FIX-CORE] cascade rack {rack} → {final_thermal}")

            else:
                final_thermal = f"increase_cooling({rack})"
                final_load    = None
                decision_source = "PREEMPTIVE_COOL"

                debug_log(f"[FIX-CORE] cascade on rack {rack} → cooling")

    # P1: Hard mode — rack 0 is hottest, fan broken, still has load.
    # Physics: rack 0 gets +3.0C/step from fan failure — must evacuate jobs.
    elif fan and hottest == 0 and temps[0] > 68.0 and loads[0] > 0.15:

        if temps[0] > 90.0:

            if loads[0] > 0.25:
                # keep evacuating aggressively
                if loads[coolest] < 0.95:
                    final_thermal   = f"migrate_jobs(0,{coolest})"
                    final_load      = f"migrate_jobs(0,{coolest})"
                    decision_source = "P1_hard_force_evac"

                    debug_log(f"[FIX-FINAL] force evac → (0,{coolest}) load0={loads[0]:.2f}")

                else:
                    # no space → create space
                    non0_heaviest = max((i for i in range(n) if i != 0), key=lambda i: loads[i])

                    final_thermal   = f"decrease_load({non0_heaviest})"
                    final_load      = None
                    decision_source = "P1_hard_make_space"

                    debug_log(f"[FIX-FINAL] create space → decrease_load({non0_heaviest})")

            else:
                # only now cooling makes sense
                final_thermal   = "increase_cooling(0)"
                final_load      = None
                decision_source = "P1_hard_final_cool"

        elif loads[0] > 0.25 and loads[coolest] < 0.95:
            # normal evacuation
            final_thermal   = f"migrate_jobs(0,{coolest})"
            final_load      = f"migrate_jobs(0,{coolest})"
            decision_source = "P1_hard_evac"

        else:
            # All migration targets are full or too hot
            if temps[0] > 90.0:

                # 🚀 FORCE SPACE CREATION instead of useless cooling
                non0_heaviest = max((i for i in range(n) if i != 0), key=lambda i: loads[i])

                if loads[non0_heaviest] > 0.7:
                    final_thermal   = f"decrease_load({non0_heaviest})"
                    final_load      = None
                    decision_source = "P1_hard_make_space"

                    debug_log(f"[FIX-3] create space: decrease_load({non0_heaviest}) "
                            f"load={loads[non0_heaviest]:.2f}")

                else:
                    # last resort → force migration anyway
                    fallback_dst = min(
                        range(n),
                        key=lambda i: (loads[i], temps[i])
                    )

                    if loads[0] > 0.25 or temps[0] > 110:
                        final_thermal   = f"migrate_jobs(0,{fallback_dst})"
                        final_load      = f"migrate_jobs(0,{fallback_dst})"
                        decision_source = "P1_hard_force_evac"

                        debug_log(f"[FIX-3] force migrate → (0,{fallback_dst}) load0={loads[0]:.2f}")

                    else:
                        final_thermal   = "increase_cooling(0)"
                        final_load      = None
                        decision_source = "P1_post_evac_cooling"

                        debug_log("[FIX-FINAL-2] stop evac → cooling rack 0")
                    
            else:
                # Rack 0 not yet critical — shed the most loaded secondary rack to create room
                non0_hot = max((i for i in range(n) if i != 0), key=lambda i: temps[i])
                if loads[non0_hot] >= 0.85:
                    final_thermal   = f"decrease_load({non0_hot})"
                    final_load      = None
                    decision_source = "P1_hard_decreaseload"
                    debug_log(f"P1 hard: all full → decrease_load({non0_hot}) "
                              f"load={loads[non0_hot]:.2f}")
                else:
                    final_thermal   = f"increase_cooling({non0_hot})"
                    final_load      = None
                    decision_source = "P1_hard_fullrack"
                    debug_log(f"P1 hard: all full → cooling rack {non0_hot}")

    # P2: Absolute temperature ceiling > 92C — cool immediately.
    elif max_temp > 92.0:
        if fan:
            non0_candidates = [(i, temps[i]) for i in range(n) if i != 0]
            non0_hot        = max(non0_candidates, key=lambda x: x[1])[0]
            final_thermal   = (f"increase_cooling({hottest})"
                               if temps[0] > temps[non0_hot] + 5.0
                               else f"increase_cooling({non0_hot})")
        else:
            final_thermal = f"increase_cooling({hottest})"
        decision_source = "P2_emergency"
        debug_log(f"P2 emergency → {final_thermal}  fan={fan}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3 — LLM PRIMARY PATH
    # ─────────────────────────────────────────────────────────────────

    elif llm_conf >= 0.70:
        # [FIX-4] P4 threshold lowered 90→86C.
        # At 86C predicted the agent has budget to recover (5C cooling > 3C heat gain).
        # At 90C the cascade is already beyond what 1 step can reverse.
        if pred_max >= 86.0 and "migrate" in llm_action:
            final_thermal   = f"increase_cooling({pred_rack})"
            final_load      = p1["load"]
            decision_source = "P4_safety_cooling"
            debug_log(f"P4 safety: predicted={pred_max:.1f}C → cool({pred_rack})")
        else:
            p2: dict | None = None
            if should_use_overseer(obs, llm_conf, c_score):
                try:
                    p2 = pass2_llm(obs, p1, c_score, predicted)
                except Exception as e:
                    debug_log(f"pass2 exception: {e}")
                    p2 = None
            else:
                debug_log("Overseer skipped — conditions safe")

            if p2 and p2.get("override_thermal") and p2.get("final_thermal"):
                p2_action = p2["final_thermal"]
                # [FIX-E] Redirect overseer to the actual predicted-hottest rack when
                # it picked a much cooler one (gap > 3C). Prevents cool(0) when rack 1
                # is 6C hotter and about to cascade.
                if "increase_cooling" in p2_action:
                    p2_rack = get_rack(p2_action)
                    if p2_rack != pred_rack and predicted[p2_rack] < predicted[pred_rack] - 3.0:
                        debug_log(f"Overseer rack corrected: {p2_action} → "
                                  f"increase_cooling({pred_rack}) "
                                  f"(pred {predicted[p2_rack]:.1f} < {predicted[pred_rack]:.1f})")
                        p2_action = f"increase_cooling({pred_rack})"
                final_thermal   = p2_action
                final_load      = p2.get("final_load") or p1["load"]
                decision_source = "P3_overseer_override"
                debug_log(f"Overseer OVERRIDE → {final_thermal}")
            else:
                final_thermal   = llm_action
                final_load      = p1["load"]
                decision_source = "LLM_primary"
                debug_log(f"LLM primary → {final_thermal} conf={llm_conf:.2f}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4 — RULE FALLBACK (LLM confidence < 0.70 or API error)
    # ─────────────────────────────────────────────────────────────────

    else:
        zone      = _tsm.state
        imbalance = max_temp - temps[overall_coolest]
        thr       = _adaptive.migrate_imbalance_threshold
        last_act  = _memory.last_action()

        debug_log(f"LLM fallback: conf={llm_conf:.2f} zone={zone} imb={imbalance:.1f}")

        if zone in (ThermalStateMachine.SAFE, ThermalStateMachine.CONTROL):
            if imbalance >= thr and loads[overall_coolest] < 0.85:
                candidate = f"migrate_jobs({hottest},{overall_coolest})"
                if candidate == last_act:
                    candidate = f"increase_cooling({hottest})"
            elif imbalance >= thr and loads[overall_coolest] >= 0.85:
                if temps[overall_coolest] < 78.0 and last_act != f"decrease_load({overall_coolest})":
                    candidate = f"decrease_load({overall_coolest})"
                else:
                    candidate = f"increase_cooling({hottest})"
            else:
                candidate = f"increase_cooling({hottest})"
        else:  # HIGH or EMERGENCY
            candidate = f"increase_cooling({hottest})"

        final_thermal   = candidate
        decision_source = f"P6_fallback(imb={imbalance:.1f})"

    # ─────────────────────────────────────────────────────────────────
    # FINALISE
    # ─────────────────────────────────────────────────────────────────

    if not final_load:
        final_load = p1["load"]
    if not final_thermal:
        final_thermal = p1["thermal"] or f"increase_cooling({hottest})"

    final_action = final_thermal or final_load or f"increase_cooling({hottest})"

    # Validate — catch hallucinated rack indices
    validated = extract_action(final_action)
    if not validated:
        p1_fallback = extract_action(p1.get("thermal", ""))
        validated   = p1_fallback or f"increase_cooling({hottest})"
        debug_log(f"Invalid action hallucinated: '{final_action}' → {validated}")

    # ─────────────────────────────────────────────────────────────────
    # POST-DECISION VETOES
    # Applied after all other logic. These are the last line of defence.
    # They catch cases where the LLM correctly interprets intent but the
    # resulting action is physically invalid or thermally dangerous.
    # ─────────────────────────────────────────────────────────────────

    # [FIX-1] Capacity + temperature veto.
    # If the migration destination has load >= 0.85, jobs are silently discarded by
    # the environment — the action costs power_cost but produces zero throughput gain.
    # If the destination temp > 80C, sending more load accelerates its cascade.
    # In both cases: redirect to cooling the hottest rack instead.
    if "migrate_jobs" in validated:
        dst = get_dest_rack(validated)

        if dst != -1:
            if task == "hard":

                # 🚀 HARD MODE SMART EVAC (NOT BLIND)
                if "migrate_jobs(0" in validated:

                    # ❌ DON'T allow dumping into overloaded rack
                    if loads[dst] >= 0.90 or temps[dst] > 90.0:

                        # 🔁 find better target
                        alt_targets = sorted(range(len(loads)), key=lambda i: (loads[i], temps[i]))

                        found = False
                        for alt in alt_targets:
                            if alt != 0 and loads[alt] < 0.90 and temps[alt] < 90:
                                validated = f"migrate_jobs(0,{alt})"
                                debug_log(f"[SMART REDIRECT] dst {dst} → {alt}")
                                found = True
                                break

                        if not found:
                            # no safe rack → create space instead
                            heavy = max([i for i in range(len(loads)) if i != 0], key=lambda i: loads[i])
                            validated = f"decrease_load({heavy})"
                            global _last_space_fix
                            _last_space_fix = heavy
                            debug_log(f"[SMART SPACE] decrease_load({heavy}) instead of bad migrate")

                    else:
                        debug_log(f"[HARD OK] evacuating → dst={dst} load={loads[dst]:.2f}")

                else:
                    # Non-critical migrations
                    if loads[dst] >= 0.95 and temps[dst] > 95:
                        validated = f"increase_cooling({hottest})"
                        debug_log(f"[HARD VETO] non-critical migration blocked dst={dst}")

            else:
                # easy/medium safe logic
                if loads[dst] >= 0.85 or temps[dst] > 80.0:
                    validated = f"increase_cooling({hottest})"
    
    if task == "hard" and fan and (loads[0] > 0.20 or temps[0] > 100):
        validated = f"migrate_jobs(0,{coolest})"
        debug_log("[FORCE] keep evacuating rack 0")
    
    # =====================================
    # 🚀 HARD MODE PRIORITY HIERARCHY
    # =====================================
    if task == "hard" and fan:

        if loads[0] > 0.10 or temps[0] > 105:
            validated = f"migrate_jobs(0,{coolest})"
            debug_log("[PRIORITY] evacuating rack 0")

        elif loads[0] > 0.05:
            validated = f"migrate_jobs(0,{coolest})"
            debug_log("[PRIORITY] finishing evacuation")

        else:
            validated = "increase_cooling(0)"
            debug_log("[PRIORITY] post-evac cooling")

            
    # [FIX-2] Thermal zone veto.
    # In EMERGENCY zone the system is beyond stable operation — migration shuffles load
    # but does not reduce the total heat being generated; cooling does.
    # In HIGH zone with max_temp > 82C the system is trending toward EMERGENCY —
    # migration at this point consistently leads to worse outcomes in all three tasks.
    if "migrate_jobs" in validated:

        # 🚀 HARD MODE: NEVER veto migration (CRITICAL)
        if task == "hard":
            debug_log("[FIX-2 HARD] migration preserved")
    
        # 🧊 EASY / MEDIUM: keep safety
        else:
            if (_tsm.state == ThermalStateMachine.EMERGENCY or
                (_tsm.state == ThermalStateMachine.HIGH and max_temp > 82.0)):
                old_act   = validated
                validated = f"increase_cooling({hottest})"
                debug_log(f"[FIX-2] Zone veto ({_tsm.state} max={max_temp:.1f}C): "
                        f"{old_act} → {validated}")

    if _last_space_fix is not None and "migrate_jobs" in validated:
        dst = get_dest_rack(validated)

        if dst == _last_space_fix:
            debug_log(f"[LOCK] avoiding immediate reuse of rack {dst}")

            # choose alternate rack (not rack 0, not locked rack)
            candidates = [i for i in range(len(loads)) if i != 0 and i != dst]
            if candidates:
                alt = min(candidates, key=lambda i: (loads[i], temps[i]))
                validated = f"migrate_jobs(0,{alt})"
    
    # Update UCB-1 bandit
    last_reward = list(_memory.rewards)[-1] if _memory.rewards else 0.0
    arm = ("migrate_jobs"   if "migrate"  in validated else
           "decrease_load"  if "decrease" in validated else
           "increase_cooling")
    _bandit.update(arm, last_reward)

    _memory.record_action(validated)
    debug_log(f"FINAL [{decision_source}] → {validated}  zone={_tsm.state}")
    return validated


# ============================================================
# RUN TASK
# ============================================================

def run_task(task: str) -> float:
    global _last_space_fix
    _last_space_fix = None
    _reset_episode_state()
    _profiler.start(task)

    steps = MAX_STEPS.get(task, 10)
    # [OUT-7] Pass task-specific step budget to env so hard mode gets 15 steps.
    env   = GreenOpsEnv(max_steps=steps)
    obs   = env.reset(task)

    log_start(task)

    _p1_last: dict = {"thermal": None, "load": None, "confidence": 0.0}

    for step in range(1, steps + 1):
        obs_snap = obs
        action   = get_action(obs, task)
        result   = env.step(action)
        reward   = result.reward if result.reward is not None else 0.0

        _memory.record_reward(reward)

        role_rewards = _decompose_reward(obs_snap, action, reward)
        log_step(step, action, reward, result.done, role_rewards)

        _profiler.record_step(
            step=step,
            obs=obs_snap,
            action=action,
            reward=reward,
            p1=_last_good_p1 or _p1_last,
            p2=None,
            thermal_state=_tsm.state,
            predicted_temps=_cascade.predict(
                list(obs_snap.rack_temp),
                list(obs_snap.cpu_load),
                getattr(obs_snap, "failed_fan", False),
            ),
        )

        obs = result.observation
        if result.done:
            break

    env.close()
    score = grade(env)
    log_end(score)
    _profiler.emit_summary(score)
    return score


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    scores: dict[str, float] = {}
    for task in ("easy", "medium", "hard"):
        scores[task] = run_task(task)

    print("\n=== FINAL SCORES ===", flush=True)
    for task, s in scores.items():
        print(f"  {task:8s}: {s:.4f}", flush=True)

    print(json.dumps({"final_scores": scores}), flush=True)


if __name__ == "__main__":
    main()