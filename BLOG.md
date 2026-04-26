# 🔥 GreenOps-X: Teaching AI to Control Data Center Systems

## 🚀 Overview

Can an AI learn to manage a data center in real time?

GreenOps-X is an AI-driven system where a model learns to control cooling and workload distribution in a simulated data center environment.

Instead of generating text, the model takes **structured control actions** like cooling racks, reducing load, and migrating jobs.

---

## 🧠 The Problem

Modern data centers face three key challenges:

* 🔥 **Overheating** → can lead to cascading failures
* ⚡ **Excessive cooling** → wastes energy
* ⚖️ **Load imbalance** → creates hotspots

Most existing systems are:

* reactive
* rule-based
* inefficient under failures

---

## ⚙️ What We Built

We designed a **physics-based simulation environment** where an AI agent learns to make control decisions.

The environment models:

* heat generation from CPU load
* temperature dynamics over time
* failure scenarios like broken fans

---

## 🎮 Action Space

The agent can take three types of actions:

* `increase_cooling(rack)`
* `decrease_load(rack)`
* `migrate_jobs(src, dst)`

Each action has trade-offs between temperature, cost, and performance.

---

## 🔑 Key Design Features

### 1. 🔮 Early Prediction (Cascade Awareness)

The system can anticipate when temperatures will rise in the next few steps.

Instead of reacting late, the agent learns to:

👉 act early and prevent failures

---

### 2. 🧠 Planning Instead of Reacting

Because of predictive signals, the model learns:

* to migrate jobs before overheating
* to balance long-term trade-offs

---

### 3. 🤖 Two-Pass Control System

We use a simple but powerful structure:

* **Actor** → proposes actions
* **Overseer** → validates safety

This avoids unsafe decisions and improves stability.

---

### 4. 🔐 Action Audit Trail

Every action is:

* hashed (SHA-256)
* stored in an append-only log

This ensures:

* traceability
* reproducibility
* trust in evaluation

---

### 5. 📊 SLA-Based Evaluation

We map scores to human-readable levels:

* Platinum → production-ready
* Gold → reliable
* Bronze → recoverable
* Breach → failure

---

## 📊 Training Approach

Instead of reinforcement learning, we used:

👉 **Supervised Fine-Tuning (SFT)** with LoRA

We generated ~4000+ samples from the environment and trained two models:

* Actor (decision-making)
* Overseer (safety validation)

---

## 📉 Training Evidence

Both models show stable convergence:

* consistent loss reduction
* training ≈ validation (good generalization)

This confirms the model is learning meaningful behavior.

---

## 📈 Results (Before vs After Training)

Before training:

* overuse of cooling
* inefficient decisions
* poor handling of failures

After training:

* better balance between cooling and migration
* lower energy usage
* improved stability in difficult scenarios

---

## 🌐 Live Demo

👉 https://adit555-green-ops-lite.hf.space/ui

You can:

* run the simulation
* observe temperature changes
* see how the agent reacts in real time

---

## 🧠 Key Insight

> Even without reinforcement learning, LLMs can learn meaningful control strategies when trained on structured environment data.

---

## 🚀 Future Work

* reinforcement learning for long-term optimization
* stronger safety policies
* scaling to larger systems

---

## 🔗 Resources

* Demo: https://adit555-green-ops-lite.hf.space/ui
* Dashboard: https://huggingface.co/spaces/Adit555/greenops-analysis-dashboard
* Repo: https://github.com/aditcool-dev/green-ops-lite

---

## 💯 Final Thought

GreenOps-X shows that LLMs can move beyond language tasks and begin to operate as **decision-making systems in real-world environments**.
