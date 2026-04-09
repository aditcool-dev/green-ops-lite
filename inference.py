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

MAX_STEPS = 10

def extract_action(action_text):
    pattern = r"(increase_cooling\(\d\)|decrease_load\(\d\)|migrate_jobs\(\d,\s*\d\))"
    match = re.search(pattern, action_text)

    if match:
        return match.group(0)

    return None


def get_action(observation):
    global last_actions
    
    prompt = f"""
You are an AI Site Reliability Engineer managing a data center with 3 server racks.

Current State:
Rack 0 → Temp: {observation.rack_temp[0]:.1f}°C, Load: {observation.cpu_load[0]:.2f}
Rack 1 → Temp: {observation.rack_temp[1]:.1f}°C, Load: {observation.cpu_load[1]:.2f}
Rack 2 → Temp: {observation.rack_temp[2]:.1f}°C, Load: {observation.cpu_load[2]:.2f}

Physics & Rules:
1. High CPU load generates heat. Temp > 75°C causes exponential thermal runaway.
2. increase_cooling(x) drops temperature but WASTES power (lowers efficiency score).
3. decrease_load(x) drops temperature but DESTROYS throughput (lowers throughput score).
4. migrate_jobs(source, target) moves load from a hot rack to a cool rack. This saves power AND maintains throughput. This is your best tool.

Instructions:
- If a rack is critically hot (>75°C), use increase_cooling(x).
- If Rack 0 is broken, use migrate_jobs(0, safe_rack).
- To manage normal heat, use migrate_jobs to move load from the hottest rack to the coolest rack.
- Only use decrease_load(x) if a rack's load is extremely high (>0.85).

Available Actions:
increase_cooling(0), increase_cooling(1), increase_cooling(2)
decrease_load(0), decrease_load(1), decrease_load(2)
migrate_jobs(0,1), migrate_jobs(1,2), migrate_jobs(2,0)

IMPORTANT: Think step-by-step internally, but output ONLY ONE valid action.
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

    last_actions.append(action)
    parsed = extract_action(action)

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
    global last_actions
    last_actions = []
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