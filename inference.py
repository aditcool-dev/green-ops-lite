import os
import re
from openai import OpenAI
from env.environment import GreenOpsEnv
from env.grader import grade

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

last_actions = []
last_rewards = []

MAX_STEPS = 10

def extract_action(action_text):
    pattern = r"(increase_cooling\(\d\)|decrease_load\(\d\)|migrate_jobs\(\d,\s*\d\))"
    match = re.search(pattern, action_text)

    if match:
        return match.group(0)

    return None


def get_action(observation):
    global last_actions, last_rewards
    
    recent_actions = last_actions[-3:] if len(last_actions) >= 3 else last_actions
    recent_rewards = last_rewards[-3:] if len(last_rewards) >= 3 else last_rewards

    actions_str = ", ".join(recent_actions) if recent_actions else "None"
    rewards_str = ", ".join([f"{r:.2f}" for r in recent_rewards]) if recent_rewards else "None"

    prompt = f"""
You are an expert AI Site Reliability Engineer managing a data center with 3 server racks.

Your goal is to:
- Prevent overheating (>75°C)
- Maintain system stability
- Minimize power usage
- Avoid cascading failures

--------------------------------------------------
Current State:

Rack 0 → Temp: {observation.rack_temp[0]:.1f}°C, Load: {observation.cpu_load[0]:.2f}
Rack 1 → Temp: {observation.rack_temp[1]:.1f}°C, Load: {observation.cpu_load[1]:.2f}
Rack 2 → Temp: {observation.rack_temp[2]:.1f}°C, Load: {observation.cpu_load[2]:.2f}

Fan Failure Status:
- failed_fan: {observation.failed_fan}

IMPORTANT:
- If failed_fan = 0 → Rack 0 cooling is permanently degraded
- Cooling Rack 0 is less effective
- Heat from Rack 0 spreads → cascading failure risk
- PRIORITY: move load away from Rack 0 early

--------------------------------------------------
Recent History:

Previous Actions:
{actions_str}

Previous Rewards:
{rewards_str}

--------------------------------------------------
Physics & Rules:

1. High CPU load generates heat
2. Temp > 75°C → exponential thermal runaway
3. increase_cooling(x):
   - Reduces temperature
   - Costs energy (efficiency penalty)
4. decrease_load(x):
   - Reduces temperature
   - Reduces throughput (penalty)
5. migrate_jobs(source, target):
   - Moves load from hot → cool rack
   - BEST for efficiency + stability

--------------------------------------------------
Strategy Guidelines:

- Always consider the FULL system, not just one rack

- EARLY WARNING:
  If any rack > 85°C:
  → Take PREEMPTIVE action using increase_cooling(x)

- EMERGENCY:
  If any rack > 90°C:
  → IMMEDIATE ACTION: increase_cooling(x)
  → Do NOT continue migrating

- If failed_fan = 0:
  → Move load away from Rack 0 as early as possible
  → Avoid relying on cooling Rack 0

- Prefer migrate_jobs over increase_cooling when safe

- Move load from HOTTEST rack → COOLEST rack

- If recent rewards are decreasing:
  → Current strategy is failing
  → SWITCH strategy immediately

- If temperatures continue rising after migration:
  → STOP migrating and use cooling

- Use decrease_load ONLY if load > 0.85 AND no better option exists

--------------------------------------------------
Anti-Repetition Rule:

- DO NOT repeat the same action more than 2 times
- If repeated actions do not improve reward → MUST change strategy

--------------------------------------------------
Available Actions:

increase_cooling(0), increase_cooling(1), increase_cooling(2)
decrease_load(0), decrease_load(1), decrease_load(2)
migrate_jobs(0,1), migrate_jobs(1,2), migrate_jobs(2,0)

--------------------------------------------------

IMPORTANT:
- Think step-by-step internally
- Output ONLY ONE valid action
- Do NOT explain
- Output must EXACTLY match one of the available actions

Action:
""" 
    
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.1,
        )
        action = res.choices[0].message.content.strip()
    except Exception as e:
        action = ""

    parsed = extract_action(action)
    if parsed:
        last_actions.append(parsed)
    
    # --- UPGRADED FALLBACK LOGIC ---
    temps = observation.rack_temp
    loads = observation.cpu_load
    is_hard = getattr(observation, "failed_fan", False)

    max_temp = max(temps)
    max_temp_idx = temps.index(max_temp)
    safe_targets = [i for i in range(3) if temps[i] < 70]
    target = min(safe_targets, key=lambda i: temps[i]) if safe_targets else temps.index(min(temps))

    # USE LLM ACTION (If valid)
    if parsed:
        return parsed

    # 2. OVERRIDE / FALLBACK LOGIC
    if is_hard and temps[0] > 70.0 and loads[0] > 0.1 and target != 0:
        return f"migrate_jobs(0,{target})"
        
    if max_temp > 75.0:
        return f"increase_cooling({max_temp_idx})"
        
    if max_temp > 68.0 and loads[max_temp_idx] > 0.2 and target != max_temp_idx:
        return f"migrate_jobs({max_temp_idx},{target})"
        
    max_load_idx = loads.index(max(loads))
    if max(loads) > 0.85:
        return f"decrease_load({max_load_idx})"

    return f"increase_cooling({max_temp_idx})"
    
def log_start(task, model):
    print(f"[START] task={task} env=greenops model={model}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, rewards, final_score):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={final_score:.3f} rewards={rewards_str}", flush=True)


def run_task(task):
    global last_actions, last_rewards
    last_actions = []
    last_rewards = []
    
    env = GreenOpsEnv()
    obs = env.reset(task)

    rewards = []
    steps = 0

    log_start(task, MODEL_NAME)

    for step in range(1, MAX_STEPS + 1):
        action = get_action(obs)
        result = env.step(action)

        reward = result.reward if result.reward is not None else 0.0
        rewards.append(reward)
        last_rewards.append(reward) 
        steps = step

        log_step(step, action, reward, result.done)

        obs = result.observation

        if result.done:
            break

    env.close()

    final_score = grade(env)
    success = final_score > 0.5
    log_end(success, steps, rewards, final_score)
    return final_score

def main():
    scores = {
        "easy": run_task("easy"),
        "medium": run_task("medium"),
        "hard": run_task("hard"),
    }
    
if __name__ == "__main__":
    main()