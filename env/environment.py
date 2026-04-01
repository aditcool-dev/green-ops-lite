import random
from .models import Observation, StepResult

random.seed(42)
MAX_STEPS = 10
TEMP_THRESHOLD = 70

class GreenOpsEnv:
    def __init__(self):
        random.seed(42)
        self.reset()

    def reset(self, task_name="None"):
        if task_name is None:
            task_name = "easy"
        self.task_name = task_name
        self.step_count = 0

        self.rack_temp = [75.0, 80.0, 65.0]
        self.cpu_load = [0.7, 0.9, 0.5]
        self.power_cost = 1.0
        self.failed_fan = False

        if task_name == "hard":
            self.failed_fan = True
            self.rack_temp = [82.0, 78.0, 72.0]   # NOT too extreme
            self.cpu_load = [0.85, 0.8, 0.75]

        if task_name == "medium":
            self.rack_temp = [78.0, 75.0, 70.0]
            self.cpu_load = [0.8, 0.75, 0.7]
            
        return self._get_obs()

    def _get_obs(self):
        return Observation(
            rack_temp=self.rack_temp,
            cpu_load=self.cpu_load,
            power_cost=self.power_cost,
            failed_fan=self.failed_fan,
            step_count=self.step_count,
        )

    def step(self, action_str):
        self.step_count += 1

        # ---- Apply Action ----
        if "increase_cooling" in action_str:
            rack = self._extract_index(action_str)
            if rack is not None:
                self.rack_temp[rack] -= 5
                self.power_cost += 0.2

        elif "decrease_load" in action_str:
            rack = self._extract_index(action_str)
            if rack is not None:
                self.cpu_load[rack] = max(0.1, self.cpu_load[rack] - 0.2)
                self.rack_temp[rack] -= 2

        elif "migrate_jobs" in action_str:
            src, tgt = self._extract_two_indices(action_str)
            if src is not None and tgt is not None:
                load = min(0.3, self.cpu_load[src] * 0.3)
                self.cpu_load[src] -= load
                self.cpu_load[tgt] += load

        # ---- Natural Dynamics ----
        for i in range(3):
            self.rack_temp[i] += self.cpu_load[i] * 2

        if self.failed_fan:
            self.rack_temp[0] += 3  # extra heating

        # ---- Clamp Values (VERY IMPORTANT) ----
        for i in range(3):
            self.cpu_load[i] = max(0.0, min(1.0, self.cpu_load[i]))
            self.rack_temp[i] = max(50.0, self.rack_temp[i])
        
        # ---- Reward ----
        reward = self._compute_reward()

        # ---- Done ----
        done = self.step_count >= MAX_STEPS or self._is_stable()

        return StepResult(
            observation=self._get_obs(),
            reward=reward,
            done=done,
            info={
                "temps": self.rack_temp,
                "loads": self.cpu_load,
                "power_cost": self.power_cost
            }
        )

    def _compute_reward(self):
        stability = sum(max(0, 1 - (t - 60)/20) for t in self.rack_temp) / 3
        efficiency = max(0, 1 - (self.power_cost / 2))
        uptime = 1.0 if all(t < 85 for t in self.rack_temp) else 0.0

        reward = 0.45 * stability + 0.3 * uptime + 0.25 * efficiency
        
        if self.task_name == "easy" and all(t < 65 for t in self.rack_temp):
            reward += 0.3

        if self.task_name == "hard" and all(t < 75 for t in self.rack_temp):
            reward += 0.2
        
        if all(t < 65 for t in self.rack_temp):
            reward += 0.2
    
        # penalties
        if any(t > 90 for t in self.rack_temp):
            reward -= 0.3

        reward -= 0.01 * self.step_count

        reward = max(-1, min(2, reward))
        return round(reward, 3)

    def _is_stable(self):
        return all(t < TEMP_THRESHOLD for t in self.rack_temp)

    def _extract_index(self, action_str):
        try:
            return int(action_str.split("(")[1].split(")")[0])
        except:
            return None

    def _extract_two_indices(self, action_str):
        try:
            nums = action_str.split("(")[1].split(")")[0].split(",")
            return int(nums[0]), int(nums[1])
        except:
            return None, None

    def state(self):
        return {
            "temp": self.rack_temp,
            "load": self.cpu_load
        }

    def close(self):
        pass