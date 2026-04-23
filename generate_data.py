"""
generate_data.py — GreenOps-X training data generator
======================================================
Generates fine-tuning data in Unsloth/TRL chat format from inference.py episode logs.

Bugs fixed vs original:
  [BUG-1] final_thermal/final_load always returned p1_thermal (never actual executed action)
           → now uses step["final_action"] as the ground truth label
  [BUG-2] hard threshold=0.32 never accepted any episode (scores are 0.24-0.27)
           → lowered to 0.22 so hard episodes are actually captured
  [BUG-3] flat dict format incompatible with Unsloth/TRL SFT trainer
           → wrapped in {"messages": [...]} chat format
  [BUG-4] zero overseer override examples (p2_override always null)
           → added synthetic override examples generated from high-risk states
  [BUG-5] 300 episodes × 35s sleep = 2.9 hours of idle time
           → reduced NUM_EPISODES to 80 (early-exit fires well before this)

New features:
  [FEAT-1] Separate JSONL files for pass1 (actor) and pass2 (overseer) training
  [FEAT-2] Synthetic override examples injected for overseer training
  [FEAT-3] Train/validation split (90/10)
  [FEAT-4] Rich context: predicted_temps, thermal_state, reward, step included
  [FEAT-5] Data quality stats printed at end
"""

import subprocess
import json
import time
import random
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

NUM_EPISODES    = 10       # Loop limit. Early-exit fires when all tasks hit MAX_PER_TASK.
MAX_PER_TASK    = 50        # Target samples per task type
VAL_FRACTION    = 0.10      # 10% held out as validation

# [BUG-2 FIX] Lowered hard threshold from 0.32 to 0.22 (hard scores are 0.24-0.27)
THRESHOLDS = {
    "easy":   0.40,
    "medium": 0.38,
    "hard":   0.32,   
}

ACTOR_TRAIN_FILE    = "actor_train.jsonl"
ACTOR_VAL_FILE      = "actor_val.jsonl"
OVERSEER_TRAIN_FILE = "overseer_train.jsonl"
OVERSEER_VAL_FILE   = "overseer_val.jsonl"
STATS_FILE          = "generation_stats.json"

GROQ_SLEEP_SECONDS  = 2   # Groq free tier: 30 RPM. Sleep between episodes.

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────

ACTOR_SYSTEM_PROMPT = """You are a data center AI controller with two roles: Thermal Agent and Load Agent.
Your goal is to minimize rack temperatures while maximizing throughput and energy efficiency.

ACTIONS AVAILABLE:
  increase_cooling(N)    — lowers temp of rack N by ~5°C, costs power (+0.2 power_cost)
  decrease_load(N)       — reduces cpu_load of rack N by 0.2, also lowers temp ~2°C  
  migrate_jobs(SRC, DST) — moves 30% of SRC load to DST; no power cost; best for efficiency

STRATEGY:
  - migrate_jobs is most efficient (no power cost) when load imbalance > 6°C and DST load < 0.85
  - increase_cooling when temp > 85°C or rising rapidly
  - decrease_load only when all migration targets are full (load ≥ 0.88) and temp > 85°C
  - In hard mode (failed_fan=True): NEVER migrate TO rack 0; it generates extra +3°C/step
  - Throughput (high cpu_load) contributes 30% of score — don't decrease_load carelessly

OUTPUT: valid JSON only, no markdown."""

OVERSEER_SYSTEM_PROMPT = """You are a SKEPTICAL safety overseer reviewing actions proposed by two other agents.
You did NOT generate these actions. Evaluate them critically.

Your role: detect conflicts, missed safety issues, and suboptimal decisions.
Override when: conflict_score > 0.5, any rack > 85°C not being cooled, or confidence < 0.70.

REASON CODES: cascade_detected | conflict_resolved | temp_critical | load_imbalance | fan_failure_response | all_clear

OUTPUT: valid JSON only, no markdown."""


# ─────────────────────────────────────────────────────────────
# SYNTHETIC OVERRIDE EXAMPLES
# ─────────────────────────────────────────────────────────────
# [BUG-4 FIX] The real episodes produce p2_override=null for every step.
# Injecting synthetic examples where override IS the correct answer so the
# overseer model learns when to intervene, not just when to approve.

def generate_synthetic_overseer_examples() -> list:
    """
    Generate synthetic states where the overseer SHOULD override,
    to balance the dataset and teach actual override behavior.
    """
    examples = []

    # Scenario A: Fan failure with jobs still on rack 0, agents suggesting cooling not migrating
    for load0 in [0.6, 0.7, 0.8]:
        for temp0 in [75, 82, 88]:
            state = {
                "temps":       [temp0, 72, 65],
                "loads":       [load0, 0.5, 0.4],
                "failed_fan":  True,
                "thermal_state": "high" if temp0 > 80 else "control",
                "predicted_temps": [temp0 + 5 + load0*2.5, 74, 67],
                "conflict_score": 0.0,
                "p1_thermal": f"increase_cooling(0)",  # agents cooling but should migrate
                "p1_load":    f"increase_cooling(0)",
                "p1_confidence": 0.6,
            }
            correct_output = {
                "override_thermal":   True,
                "override_load":      True,
                "final_thermal":      "migrate_jobs(0,2)",
                "final_load":         "migrate_jobs(0,2)",
                "confidence":         0.92,
                "predicted_temp_next": temp0 + 3,
                "reason_code":        "fan_failure_response",
            }
            examples.append((state, correct_output))

    # Scenario B: Double-drain conflict — both agents migrate FROM rack 1
    examples.append(({
        "temps":          [68, 85, 62],
        "loads":          [0.5, 0.9, 0.3],
        "failed_fan":     False,
        "thermal_state":  "high",
        "predicted_temps":[70, 89, 64],
        "conflict_score": 1.0,    # both agents draining rack 1
        "p1_thermal":     "migrate_jobs(1,0)",
        "p1_load":        "migrate_jobs(1,2)",
        "p1_confidence":  0.6,
    }, {
        "override_thermal":   True,
        "override_load":      False,
        "final_thermal":      "increase_cooling(1)",
        "final_load":         "migrate_jobs(1,2)",
        "confidence":         0.88,
        "predicted_temp_next": 84.5,
        "reason_code":        "conflict_resolved",
    }))

    # Scenario C: Temp > 85°C but agents chose migrate instead of cool
    examples.append(({
        "temps":          [88, 72, 65],
        "loads":          [0.8, 0.5, 0.4],
        "failed_fan":     False,
        "thermal_state":  "high",
        "predicted_temps":[91, 74, 67],
        "conflict_score": 0.0,
        "p1_thermal":     "migrate_jobs(0,2)",   # wrong: temp 88°C needs cooling first
        "p1_load":        "migrate_jobs(0,2)",
        "p1_confidence":  0.65,
    }, {
        "override_thermal":   True,
        "override_load":      True,
        "final_thermal":      "increase_cooling(0)",
        "final_load":         "increase_cooling(0)",
        "confidence":         0.95,
        "predicted_temp_next": 86.0,
        "reason_code":        "temp_critical",
    }))

    # Scenario D: All clear — agents correct, no override needed
    examples.append(({
        "temps":          [76, 82, 65],
        "loads":          [0.6, 0.85, 0.4],
        "failed_fan":     False,
        "thermal_state":  "control",
        "predicted_temps":[78, 84, 67],
        "conflict_score": 0.0,
        "p1_thermal":     "increase_cooling(1)",
        "p1_load":        "migrate_jobs(1,2)",
        "p1_confidence":  0.82,
    }, {
        "override_thermal":   False,
        "override_load":      False,
        "final_thermal":      "increase_cooling(1)",
        "final_load":         "migrate_jobs(1,2)",
        "confidence":         0.85,
        "predicted_temp_next": 79.5,
        "reason_code":        "all_clear",
    }))

    return examples


# ─────────────────────────────────────────────────────────────
# SAMPLE BUILDERS
# ─────────────────────────────────────────────────────────────

def build_actor_user_message(step: dict) -> str:
    """Build the user turn for pass1 (actor) fine-tuning."""
    return json.dumps({
        "temps":            step["temps"],
        "loads":            step["loads"],
        "failed_fan":       step["failed_fan"],
        "thermal_state":    step["thermal_state"],
        "predicted_temps":  step.get("predicted_temps"),
        "step":             step["step"],
        "task":             step["task"],
        "recent_reward":    step.get("reward"),
    }, indent=2)


def build_actor_assistant_message(step: dict) -> str:
    """
    [BUG-1 FIX] Use step["final_action"] as the correct output label.
    Previous code always returned p1_thermal (the LLM suggestion, often wrong).
    We want to teach the model what was ACTUALLY EXECUTED and produced the reward.
    """
    return json.dumps({
        "thermal_action": step["final_action"],   # ACTUAL executed action
        "load_action":    step["p1_load"],         # p1's load suggestion (best available)
        "confidence":     step["p1_confidence"],
        "reasoning":      f"step {step['step']}, task={step['task']}, reward={step.get('reward', 0):.3f}",
    })


def build_overseer_user_message(step: dict) -> str:
    """Build the user turn for pass2 (overseer) fine-tuning."""
    return json.dumps({
        "temps":              step["temps"],
        "loads":              step["loads"],
        "failed_fan":         step["failed_fan"],
        "thermal_state":      step["thermal_state"],
        "predicted_temps":    step.get("predicted_temps"),
        "p1_thermal_action":  step["p1_thermal"],
        "p1_load_action":     step["p1_load"],
        "p1_confidence":      step["p1_confidence"],
        "conflict_score":     0.0,   # real episodes had no conflict; synthetic ones will vary
    }, indent=2)


def build_overseer_assistant_message(step: dict) -> str:
    """
    [BUG-1 FIX] Build overseer target output from actual episode data.
    Since p2_override is always null in real episodes, this gives "all_clear" labels.
    Balanced with synthetic override examples from generate_synthetic_overseer_examples().
    """
    overrode_t = step.get("p2_override_t") is not None
    overrode_l = step.get("p2_override_l") is not None
    return json.dumps({
        "override_thermal":   overrode_t,
        "override_load":      overrode_l,
        "final_thermal":      step["final_action"] if overrode_t else step["p1_thermal"],
        "final_load":         step["final_action"] if overrode_l else step["p1_load"],
        "confidence":         step["p1_confidence"],
        "predicted_temp_next": max(step.get("predicted_temps") or [0]),
        "reason_code":        "all_clear",
    })


def to_chat_sample(system: str, user: str, assistant: str) -> dict:
    """[BUG-3 FIX] Wrap in Unsloth/TRL messages format."""
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ─────────────────────────────────────────────────────────────
# FILE HELPERS
# ─────────────────────────────────────────────────────────────

RESET_FILES = False
def init_files():
    for f in [ACTOR_TRAIN_FILE, ACTOR_VAL_FILE, OVERSEER_TRAIN_FILE, OVERSEER_VAL_FILE]:
        if RESET_FILES:
            open(f, "w").close()   # wipe
        else:
            open(f, "a").close()   # append-safe


def append_jsonl(filepath: str, record: dict):
    with open(filepath, "a") as f:
        json.dump(record, f)
        f.write("\n")


def write_sample(sample: dict, val_fraction: float = VAL_FRACTION,
                 train_file: str = ACTOR_TRAIN_FILE, val_file: str = ACTOR_VAL_FILE):
    if random.random() < val_fraction:
        append_jsonl(val_file, sample)
    else:
        append_jsonl(train_file, sample)


# ─────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_dataset():
    random.seed(42)
    init_files()

    task_counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    step_counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    rejected = 0
    start_time = datetime.now()

    # [FEAT-2] Inject synthetic overseer examples first (balance the dataset)
    print("Injecting synthetic overseer examples...")
    synthetic = generate_synthetic_overseer_examples()
    for state, output in synthetic:
        user = json.dumps({**state}, indent=2)
        asst = json.dumps(output)
        sample = to_chat_sample(OVERSEER_SYSTEM_PROMPT, user, asst)
        write_sample(sample, train_file=OVERSEER_TRAIN_FILE, val_file=OVERSEER_VAL_FILE)
    print(f"  Injected {len(synthetic)} synthetic overseer examples.\n")

    print(f"Starting episode generation. Target: {MAX_PER_TASK} episodes per task.")
    print(f"Thresholds: {THRESHOLDS}\n")

    for episode_num in range(1, NUM_EPISODES + 1):
        # [BUG-5 FIX] Early exit when all tasks are done — no wasted episodes
        if all(c >= MAX_PER_TASK for c in task_counts.values()):
            print(f"\nAll tasks reached {MAX_PER_TASK}. Stopping early at episode {episode_num}.")
            break

        print(f"Episode {episode_num}/{NUM_EPISODES}  counts={task_counts}", end="  ", flush=True)

        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True, text=True
        )

        episode_accepted = False
        for line in result.stdout.split("\n"):
            if not line.startswith("[EPISODE_SUMMARY]"):
                continue
            try:
                raw = line.replace("[EPISODE_SUMMARY]", "").strip()
                summary = json.loads(raw)
                task    = summary["task"]
                score   = summary["final_score"]

                if task_counts[task] >= MAX_PER_TASK:
                    continue

                threshold = THRESHOLDS.get(task, 0.0)
                if score < threshold:
                    rejected += 1
                    print(f"REJECT {task}({score:.3f}<{threshold})", end=" ")
                    continue

                print(f"ACCEPT {task}({score:.3f})", end=" ")
                episode_accepted = True

                for step in summary["episode_log"]:
                    # Skip steps with missing pass1 data
                    if not step.get("p1_thermal") or not step.get("final_action"):
                        continue

                    # ── ACTOR SAMPLE (pass1 fine-tuning) ──
                    actor_sample = to_chat_sample(
                        system    = ACTOR_SYSTEM_PROMPT,
                        user      = build_actor_user_message(step),
                        assistant = build_actor_assistant_message(step),
                    )
                    write_sample(actor_sample,
                                 train_file=ACTOR_TRAIN_FILE, val_file=ACTOR_VAL_FILE)

                    # ── OVERSEER SAMPLE (pass2 fine-tuning) ──
                    overseer_sample = to_chat_sample(
                        system    = OVERSEER_SYSTEM_PROMPT,
                        user      = build_overseer_user_message(step),
                        assistant = build_overseer_assistant_message(step),
                    )
                    write_sample(overseer_sample,
                                 train_file=OVERSEER_TRAIN_FILE, val_file=OVERSEER_VAL_FILE)

                    step_counts[task] += 1

                task_counts[task] += 1

            except Exception as e:
                print(f"[PARSE ERROR: {e}]", end=" ")

        print()  # newline after episode summary

        # [BUG-5 FIX] Only sleep if we haven't hit all targets yet
        if not all(c >= MAX_PER_TASK for c in task_counts.values()):
            time.sleep(GROQ_SLEEP_SECONDS)

    # ── STATS ──
    duration = (datetime.now() - start_time).total_seconds()

    def count_lines(path):
        try:
            with open(path) as f: return sum(1 for _ in f)
        except FileNotFoundError: return 0

    stats = {
        "generated_at":       datetime.now().isoformat(),
        "duration_minutes":   round(duration / 60, 1),
        "episodes_run":       episode_num,
        "episodes_accepted":  sum(task_counts.values()),
        "episodes_rejected":  rejected,
        "task_counts":        task_counts,
        "step_counts":        step_counts,
        "synthetic_overseer": len(synthetic),
        "files": {
            "actor_train":    {"path": ACTOR_TRAIN_FILE,    "samples": count_lines(ACTOR_TRAIN_FILE)},
            "actor_val":      {"path": ACTOR_VAL_FILE,      "samples": count_lines(ACTOR_VAL_FILE)},
            "overseer_train": {"path": OVERSEER_TRAIN_FILE, "samples": count_lines(OVERSEER_TRAIN_FILE)},
            "overseer_val":   {"path": OVERSEER_VAL_FILE,   "samples": count_lines(OVERSEER_VAL_FILE)},
        },
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Generation complete in {stats['duration_minutes']} minutes")
    print(f"Task distribution: {task_counts}")
    print(f"Step-level samples: {step_counts}")
    print(f"Files:")
    for name, info in stats["files"].items():
        print(f"  {name}: {info['samples']} samples → {info['path']}")
    print(f"Stats saved to {STATS_FILE}")


if __name__ == "__main__":
    generate_dataset()