from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# SLA tier definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SLATier:
    name:          str
    label:         str          # display label
    color:         str          # hex for UI
    icon:          str
    min_score:     float        # minimum grader score required
    max_temp:      float        # no rack may have exceeded this
    max_power:     float        # power_cost ceiling at episode end
    description:   str


TIERS = [
    SLATier(
        name        = "PLATINUM",
        label       = "Platinum",
        color       = "#00ffc8",
        icon        = "◆",
        min_score   = 0.72,
        max_temp    = 82.0,
        max_power   = 1.6,
        description = "Production-ready. Agent fully controls thermal dynamics "
                      "with no human intervention required.",
    ),
    SLATier(
        name        = "GOLD",
        label       = "Gold",
        color       = "#FFD700",
        icon        = "◈",
        min_score   = 0.55,
        max_temp    = 88.0,
        max_power   = 1.9,
        description = "Acceptable operation. Occasional manual review recommended "
                      "for edge cases and hardware failures.",
    ),
    SLATier(
        name        = "SILVER",
        label       = "Silver",
        color       = "#ffb400",
        icon        = "◇",
        min_score   = 0.38,
        max_temp    = 93.0,
        max_power   = 2.3,
        description = "Degraded performance. Manual operator intervention required "
                      "for hard-mode scenarios.",
    ),
    SLATier(
        name        = "BREACH",
        label       = "SLA Breach",
        color       = "#ff3250",
        icon        = "✕",
        min_score   = 0.0,
        max_temp    = 999.0,
        max_power   = 999.0,
        description = "SLA violated. Critical thermal event or unacceptable score. "
                      "Incident ticket auto-filed.",
    ),
]

# Pre/post comparison anchors — your real measured scores
BASELINE_SCORES = {
    "easy":   {"score": 0.41, "tier": "SILVER"},
    "medium": {"score": 0.40, "tier": "SILVER"},
    "hard":   {"score": 0.36, "tier": "SILVER"},
}
POST_TUNING_SCORES = {
    "easy":   {"score": 0.44, "tier": "SILVER"},
    "medium": {"score": 0.41, "tier": "SILVER"},
    "hard":   {"score": 0.38, "tier": "SILVER"},
}


# ──────────────────────────────────────────────────────────────────────────────
# SLA violation events
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SLAEvent:
    step:       int
    event_type: str    # "OVERHEAT" | "POWER_SPIKE" | "SCORE_FLOOR"
    rack:       Optional[int]
    value:      float
    threshold:  float
    message:    str


# ──────────────────────────────────────────────────────────────────────────────
# SLA Report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SLAReport:
    tier:              str
    tier_label:        str
    tier_color:        str
    tier_icon:         str
    tier_description:  str
    grade_score:       float
    peak_temp:         float
    peak_power:        float
    min_reward:        float
    avg_reward:        float
    total_steps:       int
    violations:        list
    passed_checks:     list
    failed_checks:     list
    improvement_vs_baseline: Optional[dict]  # filled if task is provided


# ──────────────────────────────────────────────────────────────────────────────
# SLAMonitor
# ──────────────────────────────────────────────────────────────────────────────

class SLAMonitor:
    """
    Tracks episode telemetry and evaluates SLA tier at episode end.
    Call record_step() each step, evaluate() at end.
    """

    # Temperature thresholds for live alerts
    ALERT_TEMP_WARNING  = 80.0
    ALERT_TEMP_CRITICAL = 90.0
    ALERT_POWER_WARNING = 1.8

    def __init__(self):
        self._steps:      list[dict]    = []
        self._events:     list[SLAEvent] = []
        self._last_report: Optional[SLAReport] = None
        self._task:       str = "easy"

    # ── recording ─────────────────────────────────────────────────────────────

    def start_episode(self, task: str = "easy"):
        self._steps.clear()
        self._events.clear()
        self._last_report = None
        self._task = task

    def record_step(
        self,
        step:       int,
        temps:      list,
        loads:      list,
        power_cost: float,
        reward:     float,
    ):
        self._steps.append({
            "step":       step,
            "temps":      list(temps),
            "loads":      list(loads),
            "power_cost": power_cost,
            "reward":     reward,
            "max_temp":   max(temps),
        })

        # Log violation events for the report
        for i, t in enumerate(temps):
            if t > self.ALERT_TEMP_CRITICAL:
                self._events.append(SLAEvent(
                    step=step, event_type="OVERHEAT", rack=i,
                    value=round(t, 1), threshold=self.ALERT_TEMP_CRITICAL,
                    message=f"Rack {i} reached {t:.1f}°C (critical threshold {self.ALERT_TEMP_CRITICAL}°C)",
                ))
        if power_cost > self.ALERT_POWER_WARNING:
            self._events.append(SLAEvent(
                step=step, event_type="POWER_SPIKE", rack=None,
                value=round(power_cost, 3), threshold=self.ALERT_POWER_WARNING,
                message=f"Power cost {power_cost:.3f} exceeded warning level {self.ALERT_POWER_WARNING}",
            ))

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, grade_score: float) -> SLAReport:
        """
        Called at episode end with the final grader score.
        Returns a full SLAReport.
        """
        if not self._steps:
            # empty episode
            return self._make_report(TIERS[-1], grade_score, 0, 0, 0, 0, 0, [], [], [])

        peak_temp  = max(s["max_temp"]   for s in self._steps)
        peak_power = max(s["power_cost"] for s in self._steps)
        rewards    = [s["reward"] for s in self._steps]
        min_rew    = min(rewards)
        avg_rew    = sum(rewards) / len(rewards)
        n_steps    = len(self._steps)

        # Determine tier
        tier = TIERS[-1]  # default to BREACH
        for t in TIERS:
            if (grade_score  >= t.min_score and
                peak_temp    <= t.max_temp  and
                peak_power   <= t.max_power):
                tier = t
                break

        # Checks
        passed, failed = [], []
        checks = [
            ("Grade score ≥ tier minimum",  grade_score  >= tier.min_score,  f"{grade_score:.4f}"),
            ("Peak temp within limit",       peak_temp    <= tier.max_temp,   f"{peak_temp:.1f}°C"),
            ("Power cost within limit",      peak_power   <= tier.max_power,  f"{peak_power:.3f}"),
            ("No critical overheating (>90°C)", peak_temp < 90.0,            f"peak={peak_temp:.1f}°C"),
            ("Positive average reward",      avg_rew > 0,                    f"avg={avg_rew:.3f}"),
        ]
        for label, ok, val in checks:
            (passed if ok else failed).append(f"{label}  [{val}]")

        # Improvement vs baseline
        improvement = None
        baseline = BASELINE_SCORES.get(self._task)
        if baseline:
            delta = grade_score - baseline["score"]
            improvement = {
                "task":            self._task,
                "baseline_score":  baseline["score"],
                "baseline_tier":   baseline["tier"],
                "current_score":   round(grade_score, 4),
                "current_tier":    tier.name,
                "delta":           round(delta, 4),
                "delta_pct":       round(delta / baseline["score"] * 100, 1),
                "improved":        delta > 0,
            }

        report = self._make_report(
            tier, grade_score, peak_temp, peak_power,
            min_rew, avg_rew, n_steps,
            [vars(e) for e in self._events], passed, failed,
        )
        report.improvement_vs_baseline = improvement
        self._last_report = report
        return report

    def _make_report(self, tier, score, peak_temp, peak_power,
                     min_rew, avg_rew, n_steps, events, passed, failed):
        return SLAReport(
            tier             = tier.name,
            tier_label       = tier.label,
            tier_color       = tier.color,
            tier_icon        = tier.icon,
            tier_description = tier.description,
            grade_score      = round(score, 4),
            peak_temp        = round(peak_temp, 1),
            peak_power       = round(peak_power, 3),
            min_reward       = round(min_rew, 3),
            avg_reward       = round(avg_rew, 3),
            total_steps      = n_steps,
            violations       = events,
            passed_checks    = passed,
            failed_checks    = failed,
            improvement_vs_baseline = None,
        )

    # ── live alerts ───────────────────────────────────────────────────────────

    def live_alerts(self, temps: list, power_cost: float) -> list[dict]:
        """
        Called every step for real-time alert panel in the UI.
        Returns a list of active alert dicts.
        """
        alerts = []
        for i, t in enumerate(temps):
            if t >= self.ALERT_TEMP_CRITICAL:
                alerts.append({
                    "level":   "CRITICAL",
                    "color":   "#ff3250",
                    "message": f"Rack {i}: {t:.1f}°C — EMERGENCY COOLING REQUIRED",
                })
            elif t >= self.ALERT_TEMP_WARNING:
                alerts.append({
                    "level":   "WARNING",
                    "color":   "#ffb400",
                    "message": f"Rack {i}: {t:.1f}°C — approaching danger zone",
                })
        if power_cost >= self.ALERT_POWER_WARNING:
            alerts.append({
                "level":   "WARNING",
                "color":   "#ffb400",
                "message": f"Power cost {power_cost:.3f} — efficiency penalty imminent",
            })
        return alerts

    # ── current state ─────────────────────────────────────────────────────────

    def current_report(self) -> dict:
        """For GET /sla — returns last completed report or in-progress summary."""
        if self._last_report:
            import dataclasses
            return dataclasses.asdict(self._last_report)
        return {
            "tier":        "IN_PROGRESS",
            "tier_label":  "In Progress",
            "tier_color":  "#5090c8",
            "total_steps": len(self._steps),
            "peak_temp":   max((s["max_temp"] for s in self._steps), default=0),
            "message":     "Episode in progress — complete episode for full SLA report",
        }

    # ── comparison table ──────────────────────────────────────────────────────

    @staticmethod
    def comparison_table() -> dict:
        """
        Static before/after comparison using your real measured scores.
        Used by GET /sla/comparison.
        """
        rows = []
        for task in ["easy", "medium", "hard"]:
            pre  = BASELINE_SCORES[task]
            post = POST_TUNING_SCORES[task]

            pre_tier  = _score_to_tier(pre["score"])
            post_tier = _score_to_tier(post["score"])

            rows.append({
                "task":             task.upper(),
                "pre_score":        pre["score"],
                "pre_tier":         pre_tier.name,
                "pre_tier_color":   pre_tier.color,
                "post_score":       post["score"],
                "post_tier":        post_tier.name,
                "post_tier_color":  post_tier.color,
                "delta":            round(post["score"] - pre["score"], 3),
                "delta_pct":        round((post["score"] - pre["score"]) / pre["score"] * 100, 1),
            })
        return {"tiers": [{"name": t.name, "label": t.label, "color": t.color,
                           "min_score": t.min_score, "description": t.description}
                          for t in TIERS],
                "comparison": rows}


def _score_to_tier(score: float) -> SLATier:
    """Map a raw grader score to its SLA tier."""
    for t in TIERS:
        if score >= t.min_score:
            return t
    return TIERS[-1]
