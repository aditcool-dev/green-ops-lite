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

# 🔥 GreenOps-X: AI for Data Center Thermal Optimization

## 🚀 Overview

GreenOps-X is an AI-driven system that learns to control data center cooling and workload distribution under dynamic conditions.

Unlike traditional rule-based systems, GreenOps-X uses **large language models (LLMs)** fine-tuned on simulated environments to make intelligent control decisions in real time.

---

## 🧠 Problem Motivation

Modern data centers face three critical challenges:

* 🔥 **Thermal instability** → overheating can cause cascading failures
* ⚡ **Energy inefficiency** → excessive cooling increases operational cost
* ⚖️ **Load imbalance** → uneven workloads create hotspots

Existing systems are:

* reactive (not predictive)
* rule-based (not adaptive)
* inefficient under failure conditions

---

## ⚙️ Our Solution

GreenOps-X combines:

### 1. 🔬 Physics-Based Environment

We simulate realistic data center behavior:

* CPU load generates heat
* Temperature evolves dynamically
* Fan failures trigger thermal cascades

---

### 2. 🎮 Action Space

The system can take actions such as:

* `increase_cooling(rack)`
* `decrease_load(rack)`
* `migrate_jobs(src, dst)`

---

### 3. 🧠 Multi-Agent Control System

We use a **two-pass architecture**:

* **Actor (LLM)** → proposes actions
* **Overseer (LLM)** → enforces safety

This ensures:

* intelligent decision-making
* safe operation under extreme conditions

---

### 4. 🔌 OpenEnv MCP Compliance

The environment supports:

* `reset()`
* `step(action)`

and follows OpenEnv-compatible schema for evaluation.

---

## 🌐 Live Demo

👉 **Control Dashboard (Main System)**
https://adit555-green-ops-lite.hf.space/ui

👉 **Analysis Dashboard (Evaluation & Metrics)**
https://huggingface.co/spaces/Adit555/greenops-analysis-dashboard

---

## 📊 Results

### 🔹 Pre vs Post Fine-Tuning (Averaged)

| Task   | Pre-Training | Post-Training | Δ Improvement |
| ------ | ------------ | ------------- | ------------- |
| EASY   | 0.41         | **0.44**      | +0.03         |
| MEDIUM | 0.40         | **0.41**      | +0.01         |
| HARD   | 0.36         | **0.38**      | +0.02         |

---

### 🔍 Key Observations

* ✅ Significant improvement in **easy scenarios**
* ✅ Stable gains in **medium complexity environments**
* ✅ Improved resilience in **hard scenarios with failures**

---

### ⚠️ Note on Evaluation

Results are averaged across multiple runs due to:

* stochastic policy sampling
* dynamic environment behavior

This ensures **robust performance trends**, not single-run noise.

---

## 📦 Project Structure

```text
green-ops-lite/
│
├── server/                 # FastAPI backend + environment
├── inference.py           # Actor + Overseer inference pipeline
├── greenops_lora_training.ipynb  # Model training notebook
├── app.py                 # UI / entry point
├── requirements.txt       # Dependencies
└── README.md
```

---

## 🤖 Model Training

We use:

* Unsloth (fast LLM training)
* LoRA fine-tuning
* ~4000+ generated samples

Goal:

```text
state → optimal action
```

---

## 🎥 Additional Resources

* 📄 Blog Post: https://huggingface.co/spaces/Adit555/green-ops-lite/blob/main/BLOG.md
* 📊 Pre/Post Training Grade Chart

![Dashboard Screenshot](images/screenshot.png)

---

## 🔥 Key Insight

> Even lightweight fine-tuning enables LLMs to learn control strategies for complex physical systems.

---

## 🚀 Future Work

* reinforcement learning (RL fine-tuning)
* stronger safety policies (overseer improvements)
* multi-agent coordination
* real-world deployment

---

## 🧑‍💻 Author

Adit Rastogi
BMS College of Engineering

---
