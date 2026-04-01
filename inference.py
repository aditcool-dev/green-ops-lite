import os
import re
from openai import OpenAI
from env.environment import GreenOpsEnv

if not os.getenv("HF_TOKEN"):
    raise ValueError("HF_TOKEN not set")

last_actions = []

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN"),
)

MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 10

def extract_action(action_text):
    pattern = r"(increase_cooling\(\d\)|decrease_load\(\d\)|migrate_jobs\(\d,\d\))"
    match = re.search(pattern, action_text)

    if match:
        return match.group(0)

    return None


def get_action(observation):
    global last_actions
    
    prompt = f"""
You are managing a data center with 3 racks.

Goal:
- Keep ALL temperatures below 70°C
- If any rack exceeds 85°C → system failure
- Minimize overheating across ALL racks

Current State:
Rack 0 → Temp: {observation.rack_temp[0]}, Load: {observation.cpu_load[0]}
Rack 1 → Temp: {observation.rack_temp[1]}, Load: {observation.cpu_load[1]}
Rack 2 → Temp: {observation.rack_temp[2]}, Load: {observation.cpu_load[2]}

Instructions:
1. Identify the HOTTEST rack
2. If multiple racks are hot → balance load
3. If load is high → decrease load or migrate
4. DO NOT repeat the same action blindly

Available Actions:
increase_cooling(0), increase_cooling(1), increase_cooling(2)
decrease_load(0), decrease_load(1), decrease_load(2)
migrate_jobs(0,1), migrate_jobs(1,2), migrate_jobs(2,0)

IMPORTANT:
- Think step-by-step internally
- Output ONLY ONE valid action from above
- Do NOT explain
- If fan failure exists, rack 0 will heat faster → prioritize it.

Action:
""" 
    
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        action = res.choices[0].message.content.strip()
    except Exception as e:
        action = ""

    
    if action in last_actions[-3:]:
        # force change
        action = "decrease_load(1)"
    last_actions.append(action)
            
    parsed = extract_action(action)

    # 🔥 RULE-BASED OVERRIDE (VERY IMPORTANT)
    temps = observation.rack_temp
    max_temp_idx = temps.index(max(temps))

    # If hottest rack is critical → always cool it
    if max(temps) > 75:
        return f"increase_cooling({max_temp_idx})"

    # If load high → reduce load
    loads = observation.cpu_load
    max_load_idx = loads.index(max(loads))

    if max(loads) > 0.8:
        return f"decrease_load({max_load_idx})"

    # fallback to parsed LLM output
    if parsed:
        return parsed

    return "increase_cooling(1)"
    


def run_task(task):
    global last_actions
    last_actions = []
    env = GreenOpsEnv()
    obs = env.reset(task)

    total = 0

    for _ in range(MAX_STEPS):
        action = get_action(obs)
        result = env.step(action)
        total += result.reward
        obs = result.observation
        if result.done:
            break

    env.close()
    return total


def main():
    scores = {
        "easy": run_task("easy"),
        "medium": run_task("medium"),
        "hard": run_task("hard"),
    }

    print(f"Easy: {scores['easy']:.2f}")
    print(f"Medium: {scores['medium']:.2f}")
    print(f"Hard: {scores['hard']:.2f}")
    
if __name__ == "__main__":
    main()