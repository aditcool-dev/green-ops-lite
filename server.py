from typing import Dict, Any
from env.environment import GreenOpsEnv
from env.grader import grade

try:
    from openenv.core.env_server import Environment
except ImportError:
    try:
        from openenv import Environment
    except ImportError:
        # local fallback so the file is importable without openenv installed
        class Environment:
            def __init__(self):
                pass
            def run(self):
                print("⚠️  Local fallback Environment (openenv not installed)")


class GreenOpsMCP(Environment):

    def __init__(self):
        super().__init__()
        self.env          = GreenOpsEnv(max_steps=15)
        self.current_task = "easy"

    # ── Core OpenEnv interface ────────────────────────────────────────────────

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        """
        Reset the environment for a given task.
        Called by the OpenEnv harness with no args (uses default "easy")
        or with task= kwarg for specific task evaluation.
        """
        self.current_task = task
        obs = self.env.reset(task_name=task)
        return self._format_obs(obs)

    def step(self, action: str) -> Dict[str, Any]:
        """
        Apply one action and return the next observation.
        reward and done are embedded in the returned dict so the
        OpenEnv framework can extract them via standard keys.
        """
        result  = self.env.step(action)
        obs     = self._format_obs(result.observation)

        # Standard keys the OpenEnv server reads for reward / termination
        obs["reward"] = float(result.reward) if result.reward is not None else 0.0
        obs["done"]   = result.done
        obs["info"]   = {
            "task":    self.current_task,
            "message": (
                "Thermal cascade active — rack 0 fan failure"
                if result.observation.failed_fan
                else "Normal operation"
            ),
        }
        return obs

    @property
    def state(self) -> Dict[str, Any]:
        """
        [FIX-2] Required third OpenEnv core method.
        Returns full current environment state — called by /state endpoint
        and by the judge harness to snapshot state between steps.
        """
        return {
            "rack_temp":   [round(t, 2) for t in self.env.rack_temp],
            "cpu_load":    [round(l, 3) for l in self.env.cpu_load],
            "power_cost":  round(self.env.power_cost, 3),
            "failed_fan":  self.env.failed_fan,
            "step_count":  self.env.step_count,
            "task":        self.current_task,
            "grade_score": round(grade(self.env), 4),
        }

    # ── Helper ────────────────────────────────────────────────────────────────

    def _format_obs(self, obs) -> Dict[str, Any]:
        """Serialize Observation pydantic model → plain dict for JSON transport."""
        return {
            "rack_temp":  [round(t, 2) for t in obs.rack_temp],
            "cpu_load":   [round(l, 3) for l in obs.cpu_load],
            "power_cost": round(obs.power_cost, 3),
            "failed_fan": getattr(obs, "failed_fan", False),
            "step_count": getattr(obs, "step_count", self.env.step_count),
        }


if __name__ == "__main__":
    print("Starting GreenOps-X OpenEnv Server...")
    server = GreenOpsMCP()
    server.run()