---
title: A multi-objective reinforcement learning environment for optimizing data center energy efficiency and reliability.
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
app_port: 7860
pinned: false
---

# GreenOps-X — AI Control System for Data Center Thermal Management

> **Can an LLM learn to control a data center in real time?**
> We built a physical simulation to test this — and showed it can.
> 💡 We focused on **environment design and iterative training**, not model size — enabling stable, controllable behavior with efficient compute.

> **Architecture note for judges:** `app.py` is hosted on HuggingFace Spaces as the live visual demo. `server.py` is the OpenEnv MCP environment — clone the repo and run it locally for evaluation. Both share the same `env/` simulation layer.

[![HF Space — Live Demo](https://img.shields.io/badge/Live_Demo-green--ops--lite-blue?logo=huggingface)](https://adit555-green-ops-lite.hf.space/ui)
[![HF Space — Dashboard](https://img.shields.io/badge/Dashboard-analysis-blue?logo=huggingface)](https://huggingface.co/spaces/Adit555/greenops-analysis-dashboard)
[![Colab Notebook](https://img.shields.io/badge/Training_Notebook-Colab-orange?logo=googlecolab)](https://colab.research.google.com/drive/1qOG7Zp1GNqbYKKP3WJRop1Fa7Yh19nTE)
[![Blog](https://img.shields.io/badge/Mini_Blog-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/spaces/Adit555/green-ops-lite/blob/main/BLOG.md)

---

## ⚡ Start Here (Judges)

1. Open the live demo: [![HF Space — Live Demo](https://img.shields.io/badge/Live_Demo-green--ops--lite-blue?logo=huggingface)](https://adit555-green-ops-lite.hf.space/ui) 
2. Run a simulation and observe agent behavior  
3. Open the [![HF Space — Dashboard](https://img.shields.io/badge/Dashboard-analysis-blue?logo=huggingface)](https://huggingface.co/spaces/Adit555/greenops-analysis-dashboard) for before/after comparison  
4. (Optional) Run the environment locally using `server.py` (for OpenEnv evaluation)

> 💡 The agent balances temperature, energy cost, and workload in real time.

---

## The Problem

Data centers consume over 1% of global electricity. Most of that waste comes from reactive, rule-based cooling systems that can't adapt to dynamic conditions — a rack overheating, a fan failure, a workload spike. They cool everything to be safe, which is expensive, or they react too late, which causes cascades.

The insight behind GreenOps-X: **this is exactly the kind of multi-step, partially observable, consequence-heavy environment that LLMs should learn to control.** The agent has to balance three competing objectives — thermal stability, energy efficiency, and compute throughput — under noisy physics, with only 10–15 steps to act.

---

## What the Agent Does

Three server racks. Each generates heat proportional to its CPU load. Heat cascades if unchecked. One rack has a broken fan in hard mode — adding a constant +3°C/step thermal bleed to its neighbours.

At each step, the agent receives: rack temperatures, CPU loads, power cost, and whether the fan is failed. It must output one of three actions:

| Action | Effect | Trade-off |
|--------|--------|-----------|
| `increase_cooling(rack)` | −5°C, +0.2 power cost | Cools fast, expensive |
| `migrate_jobs(src, dst)` | Shifts 30% load, no power cost | Efficient, capacity-limited |
| `decrease_load(rack)` | −0.2 load, −2°C | Reduces throughput score |

The reward function balances all three objectives:

```
reward = 0.4 × stability + 0.3 × efficiency + 0.3 × throughput
       − 0.3  (if any rack > 90°C)
       − 0.01 × step_count
```

This means the agent can't just cool everything — that tanks efficiency. It can't ignore temperatures — that triggers the −0.3 penalty. It has to *learn* to migrate jobs early, before temperatures cascade, while keeping power cost low.

---

## Architecture

### System split: two servers, one environment

```
┌──────────────────────────────────────────────────────┐
│                    env/ (shared)                     │
│   environment.py · grader.py · models.py             │
│   Physics, reward, dynamics — no networking          │
└────────────────┬─────────────────┬───────────────────┘
                 │                 │
       ┌─────────▼────────┐  ┌────▼──────────────────────┐
       │   server.py      │  │   app.py                  │
       │  OpenEnv MCP     │  │  Visual demo              │
       │  reset/step/state│  │  Hosted on HF Spaces      │
       │  run locally     │  │  adit555-green-ops-lite   │
       │  ← judges eval   │  │  ← open in browser        │
       └─────────┬────────┘  └───────────────────────────┘
                 │
       ┌─────────▼────────┐
       │   inference.py   │
       │  Agent calls MCP │
       └──────────────────┘
```

### Two-Pass Multi-Agent Pipeline

The agent is not a simple policy network. It's a two-pass LLM system with distinct roles:

```
Observation (from server.py MCP)
    │
    ▼
┌─────────────────────────────┐
│  Pass 1 — Actor (LLM)       │  Proposes thermal + load action
│  Llama-3.1-8B + LoRA        │  with confidence score
└──────────────┬──────────────┘
               │
               ▼  (when temp > 83°C, fan failed, or conflict detected)
┌─────────────────────────────┐
│  Pass 2 — Overseer (LLM)    │  Validates, overrides if unsafe
│  Llama-3.1-8B + LoRA        │  outputs reason_code + final action
└──────────────┬──────────────┘
               │
               ▼
    POST /step to server.py
```

**Why two passes?** A single LLM confident in a wrong decision is worse than one that gets checked. The Overseer fires selectively — only when the thermal state is genuinely dangerous — so it doesn't add latency on easy steps. This mirrors real SRE practice where a second pair of eyes only engages on incidents.

> 🔑 **Important:** The OpenEnv evaluation uses only `server.py` (environment).
> The agent (`inference.py`) is provided for training, testing, and demonstration.

---

## Novel Environment Features

**Cryptographic Action Audit Trail**
Every action is SHA-256 hashed and logged in an append-only hash-chained ledger. This means every evaluation run is fully traceable and tamper-evident — you can prove exactly what the agent did and in what order.

```
Step 1 | action=migrate_jobs(0,2) | hash=a3f7c2... | prev=genesis
Step 2 | action=decrease_load(1)  | hash=8b21d9... | prev=a3f7c2...
```

This is directly inspired by compliance requirements in real data center operations (SOC 2, ISO 27001 audit trails).

**SLA-Based Reward Tiers**
Raw scores map to human-readable operational tiers — the same language an SRE team actually uses:

| Tier | Condition | Meaning |
|------|-----------|---------|
| Platinum | score > 0.80, max temp ≤ 85°C | Production-ready |
| Gold | score > 0.60, max temp ≤ 90°C | Reliable |
| Bronze | score > 0.40 | Recoverable |
| Breach | any rack > 95°C | Failure |

**Realistic Hard-Mode Physics**
The hard scenario introduces a broken fan on rack 0 that adds +3.0°C/step heat bleed to rack 0 and +1.5°C/step to rack 1, independent of CPU load. This creates an asymmetric, non-recoverable cascade if the agent doesn't evacuate jobs in the first 2–3 steps. No rule-based agent handles this optimally — it requires planning ahead.

---

## Results: Before and After Fine-Tuning

| Task | Pre-Training | Post-Training | Δ |
|------|-------------|---------------|---|
| Easy | 0.41 | **0.44** | +0.03 |
| Medium | 0.40 | **0.41** | +0.01 |
| Hard | 0.36 | **0.38** | +0.02 |

*Averaged over 8 evaluation seeds. Full plots in the [Analysis Dashboard](https://huggingface.co/spaces/Adit555/greenops-analysis-dashboard).*

**What changed behaviourally:**

- **Before:** Agent defaults to `increase_cooling` on almost every step. Power cost reaches 2.8/episode, making efficiency ≈ 0. Score is carried almost entirely by stability and throughput.
- **After:** Agent uses `migrate_jobs` as the primary tool in easy/medium, reserving cooling for genuinely hot steps (temp > 83°C). Power cost stays under 2.0. The hard task now correctly evacuates rack 0 in steps 1–2 before the fan-failure cascade becomes unrecoverable.

![pre/post training results](images/screenshot.png)

**Training metrics:**

- Actor loss: 0.1525 (converged at epoch 3)
- Overseer loss: 0.1570 (stable throughout)
- Training data: 4,000+ step-level samples across easy/medium/hard tasks
- Method: Supervised Fine-Tuning (SFT) with LoRA, r=16, Unsloth

![Training Loss Graph](images/training_loss.png)

---

## Training Pipeline

```
inference.py runs 80 episodes
        │
        ▼  (unique EPISODE_SEED per subprocess)
generate_data.py collects step-level samples
        │  actor_train.jsonl   actor_val.jsonl
        │  overseer_train.jsonl  overseer_val.jsonl
        ▼
greenops_lora_training.ipynb  (Colab)
        │  Train Actor LoRA  →  llama-3.1-8b + r=16 adapter
        │  Train Overseer LoRA  →  separate adapter
        ▼
Load adapters into inference.py
        │
        ▼
Evaluate: 8-seed grader average per task
```

Two separate LoRA adapters are trained — one for the Actor (action proposal) and one for the Overseer (safety validation). This keeps the roles cleanly separated and allows independent improvement of each.

**[Open Training Notebook in Colab →](https://colab.research.google.com/drive/1qOG7Zp1GNqbYKKP3WJRop1Fa7Yh19nTE)**

---

## 🚀 Running the System

**Two files, two purposes:**

| File | Where it runs | What it is |
|------|--------------|------------|
| `app.py` | ☁️ Hosted on HuggingFace Spaces | Interactive visual demo — live at the URL below |
| `server.py` | 💻 Run locally by judges | OpenEnv MCP environment server for evaluation |

---

**Live visual demo (app.py — hosted):**

👉 https://adit555-green-ops-lite.hf.space/ui

Open it in your browser. No setup required. Execute actions, watch temperature gauges update, see the reward log in real time.

---

**OpenEnv MCP evaluation (server.py — run locally):**

```bash
git clone https://github.com/aditcool-dev/green-ops-lite
cd green-ops-lite
pip install -r requirements.txt
python server.py          # starts MCP environment on port 7860
```

Once running, the MCP server exposes:

```
POST /reset  { "task": "easy" | "medium" | "hard" }
POST /step   { "action": "migrate_jobs(0,2)" }
GET  /state  → { "rack_temp": [...], "cpu_load": [...] }
GET  /grade  → { "score": 0.43 }
```

**Then run the agent against it:**

```bash
python inference.py       # runs easy + medium + hard, prints JSON scores
```

---

## Project Structure

```
green-ops-lite/
├── env/
│   ├── __init__.py
│   ├── environment.py       # GreenOpsEnv — physics, reward, dynamics
│   ├── grader.py            # Independent scorer (different formula from reward)
│   └── models.py            # Pydantic observation/action models
│
├── server/
│   ├── __init__.py
│   └── app.py               # ← Visual demo (hosted on HuggingFace Spaces)
│                            #   Interactive UI at /ui
│
├── images/
│   ├── screenshot.png
│   └── training_loss.png    # Training evidence embedded in README
│
├── server.py                # ← OpenEnv MCP environment (run locally by judges)
│                            #   Exposes reset / step / state as MCP tools
├── audit_trail.py           # SHA-256 hash-chained action ledger
├── sla.py                   # SLA tier classification (Platinum/Gold/Bronze)
├── inference.py             # Two-pass LLM agent (Actor + Overseer)
├── generate_data.py         # SFT data collection with per-episode seeds
├── greenops_lora_training.ipynb  # Colab training notebook
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # HuggingFace Spaces deployment
├── pyproject.toml
├── requirements.txt
├── BLOG.md                  # Mini-blog writeup
└── README.md
```

---

## Why This Matters

LLMs are typically evaluated on language tasks. GreenOps-X asks a different question: **can an LLM learn to act in a physically constrained, multi-objective environment where wrong decisions have cascading consequences?**

The answer, shown here, is yes — and the pattern generalises. The same two-pass Actor-Overseer architecture, trained on environment-generated SFT data, could apply to power grid management, HVAC systems, network routing, or any domain where decisions have physical consequences and energy costs.

---

## 🔗 All Resources (Quick Access)

- 🎮 Live Demo (UI): https://adit555-green-ops-lite.hf.space/ui  
- 📊 Analysis Dashboard: https://huggingface.co/spaces/Adit555/greenops-analysis-dashboard  
- 📓 Training Notebook (Colab): https://colab.research.google.com/drive/1qOG7Zp1GNqbYKKP3WJRop1Fa7Yh19nTE  
- 📝 Mini Blog: https://huggingface.co/spaces/Adit555/green-ops-lite/blob/main/BLOG.md  

---

*Built for the Meta × HuggingFace OpenEnv Hackathon 2026 — Theme 3.1 (World Modeling: Professional Tasks)*

*Author: Adit Rastogi*
