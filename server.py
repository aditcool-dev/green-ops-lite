"""
GreenOps-X — server.py
=======================
OpenEnv MCP Compliant Wrapper for GreenOpsEnv
This exposes the environment to the OpenEnv evaluation harness.
"""

import json
from typing import Dict, Any, Tuple
# Note: Ensure you have openenv installed (`pip install openenv`)
from env.environment import GreenOpsEnv

try:
    from openenv.mcp import MCPEnvironment  # judge environment (real MCP)
except Exception:
    try:
        from openenv import MCPEnvironment  # if exposed directly (rare)
    except Exception:
        # local fallback (so your code runs)
        class MCPEnvironment:
            def run(self):
                print("⚠️ Local fallback MCPEnvironment running (no MCP)")

class GreenOpsMCP(MCPEnvironment):
    def __init__(self):
        super().__init__()
        # We instantiate our core environment here
        self.env = GreenOpsEnv(max_steps=15) 
        self.current_task = "easy"

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        """
        OpenEnv interface: Resets the environment for a specific task.
        """
        self.current_task = task
        obs = self.env.reset(task_name=task)
        
        return self._format_obs(obs)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        OpenEnv interface: Applies the agent's action and steps the physics forward.
        """
        result = self.env.step(action)
        
        # Convert our custom Observation object into a standard dictionary for OpenEnv
        obs_dict = self._format_obs(result.observation)
        
        # OpenEnv expects: observation, reward, done, info
        info = {
            "task": self.current_task,
            "power_cost": result.observation.power_cost,
            "message": "Thermal cascade detected" if getattr(result.observation, "failed_fan", False) else "Normal operation"
        }
        
        reward = float(result.reward) if result.reward is not None else 0.0
        
        return obs_dict, reward, result.done, info

    def _format_obs(self, obs) -> Dict[str, Any]:
        """Helper to serialize observations for the JSON-RPC interface"""
        return {
            "rack_temp": [round(t, 2) for t in obs.rack_temp],
            "cpu_load": [round(l, 3) for l in obs.cpu_load],
            "power_cost": round(obs.power_cost, 3),
            "failed_fan": getattr(obs, "failed_fan", False)
        }

if __name__ == "__main__":
    # Start the standard MCP JSON-RPC server so the OpenEnv harness can connect
    print("Starting GreenOps-X MCP Server...")
    server = GreenOpsMCP()
    server.run()