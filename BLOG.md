# 🔥 GreenOps-X: AI for Data Center Thermal Optimization

## 🚀 Overview

GreenOps-X is an AI-driven system that learns to control data center cooling and workload distribution under dynamic conditions.

It simulates:
- thermal dynamics
- hardware failures
- energy constraints

and learns optimal actions to maintain stability.

---

## 🧠 Problem

Modern data centers face:

- 🔥 overheating → cascading failures  
- ⚡ excessive cooling → high energy cost  
- ⚖️ poor load balance → hotspots  

Traditional systems are rule-based and reactive.

---

## ⚙️ Solution

GreenOps-X combines:

### 1. Physics-Based Environment

- CPU load generates heat  
- Thermal runaway beyond thresholds  
- Fan failure triggers cascade  

---

### 2. Action Space

- `increase_cooling(rack)`
- `decrease_load(rack)`
- `migrate_jobs(src, dst)`

---

### 3. Two-Pass Control (Actor + Overseer)

- Actor → proposes actions  
- Overseer → enforces safety  

This ensures stable and safe decisions.

---

### 4. OpenEnv MCP Integration

- `reset()` and `step()` implemented  
- OpenEnv-compatible schema  
- MCP-ready environment  

---

## 📊 Data Generation

We generated ~4000+ samples using:

- multi-task scenarios (easy / medium / hard)
- physics-based rollouts
- reward filtering

---

## 🤖 Model Training

We use:

- Mistral-7B (Unsloth)
- LoRA fine-tuning

Goal:

state → optimal action

---

## 📈 Results (Pre-Training)

Before training:

- repetitive decisions  
- inefficient cooling  
- slower recovery  

---

## 🔥 Expected Improvements

After fine-tuning:

- faster stabilization  
- lower power cost  
- smarter migration decisions  

---

## 🌐 Live Demo

👉 https://adit555-green-ops-lite.hf.space/ui

- Reset environment  
- Execute actions  
- Observe system behavior  

---

## 🧠 Key Insight

> AI can control complex physical systems when grounded in realistic environments.

---

## 🚀 Future Work

- reinforcement learning  
- multi-agent coordination  
- real-world deployment  

---

## 🔗 Repository

https://github.com/aditcool-dev/green-ops-lite