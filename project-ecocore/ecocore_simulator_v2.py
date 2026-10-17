
"""EcoCore Member 3 decision engine v2.

Read-only what-if simulation. It never changes real workloads.

Fixes versus the earlier version:
1. Projected trends are retained in scenario scoring instead of being forced to 0.
2. High residual risk produces "Escalate / investigate" instead of incorrectly
   recommending "No action".
3. Recommendation separates risk from intervention penalty.
"""

from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional

RiskPredictor = Callable[[dict, float, float], float]


def _bounded(value, lo, hi):
    return max(lo, min(hi, float(value)))


def _metric(metrics, key, lower, upper):
    if key not in metrics:
        raise ValueError(f"Missing telemetry key: {key}")
    value = float(metrics[key])
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{key} must be between {lower} and {upper}; got {value}")
    return value


def simulate_scenarios(
    metrics: dict,
    cpu_trend: float = 0.0,
    memory_trend: float = 0.0,
    predict_fn: Optional[RiskPredictor] = None,
    horizon_samples: int = 3,
    defer_cpu_relief: float = 15.0,
    defer_memory_relief: float = 5.0,
    redistribute_cpu_relief: float = 10.0,
    redistribute_memory_overhead: float = 2.0,
    allow_defer: bool = True,
    target_capacity_available: bool = False,
    target_compatible: bool = False,
) -> List[Dict]:
    cpu = _metric(metrics, "cpu_pct", 0, 100)
    mem = _metric(metrics, "memory_pct", 0, 100)
    load = _metric(metrics, "load_per_core", 0, 10)

    if not 1 <= int(horizon_samples) <= 12:
        raise ValueError("horizon_samples must be between 1 and 12")

    predictor = predict_fn
    if predictor is None:
        raise ValueError("A real predictor function is required for v2 scenario scoring.")

    choices = [
        ("No action", 0.0, 0.0, 1.0, 0.00, True,
         "Continue current workload with no intervention."),
        ("Defer background workload", -defer_cpu_relief, -defer_memory_relief,
         0.45, 0.04, bool(allow_defer),
         "Only non-critical background work is hypothetically deferred."),
        ("Redistribute workload", -redistribute_cpu_relief,
         redistribute_memory_overhead, 0.60, 0.06,
         bool(target_capacity_available and target_compatible),
         "Requires a confirmed compatible destination with spare capacity."),
    ]

    results = []

    for name, dcpu, dmem, trend_scale, penalty, available, explanation in choices:
        projected_cpu = _bounded(
            cpu + cpu_trend * horizon_samples * trend_scale + dcpu, 0, 100
        )
        projected_mem = _bounded(
            mem + memory_trend * horizon_samples * trend_scale + dmem, 0, 100
        )
        projected_load = _bounded(
            load + max(cpu_trend, 0) * horizon_samples * trend_scale * 0.0037
            + dcpu * 0.007,
            0,
            3,
        )

        projected_cpu_trend = cpu_trend * trend_scale
        projected_memory_trend = memory_trend * trend_scale

        projected_metrics = dict(metrics)
        projected_metrics.update({
            "cpu_pct": projected_cpu,
            "memory_pct": projected_mem,
            "load_per_core": projected_load,
            "cpu_trend": projected_cpu_trend,
            "memory_trend": projected_memory_trend,
        })

        raw_risk = float(
            predictor(
                projected_metrics,
                projected_cpu_trend,
                projected_memory_trend,
            )
        )
        if not math.isfinite(raw_risk) or not 0 <= raw_risk <= 1:
            raise ValueError("predict_fn must return a probability in [0,1]")

        results.append({
            "action": name,
            "available": available,
            "estimated_risk": raw_risk if available else raw_risk,
            "projected_cpu_pct": projected_cpu,
            "projected_memory_pct": projected_mem,
            "projected_load_per_core": projected_load,
            "projected_cpu_trend": projected_cpu_trend,
            "projected_memory_trend": projected_memory_trend,
            "intervention_penalty": penalty,
            "decision_score": raw_risk + penalty if available else None,
            "explanation": explanation,
            "risk_method": "Member 2 RandomForest predictor",
            "restriction": (
                None if available
                else "Requires confirmed compatible destination and spare capacity."
            ),
        })

    return results


def recommend(scenarios: List[Dict]) -> Dict:
    valid = [
        row for row in scenarios
        if row.get("available") and row.get("decision_score") is not None
    ]
    if not valid:
        return {
            "action": "Escalate / investigate",
            "reason": "No available intervention was verified.",
        }

    best = min(valid, key=lambda r: (r["decision_score"], r["intervention_penalty"]))

    # A decision engine should not call a critically risky state "No action"
    # merely because every available action is also risky.
    if float(best["estimated_risk"]) >= 0.90:
        return {
            **best,
            "action": "Escalate / investigate",
            "original_best_action": best["action"],
            "reason": (
                "All currently available choices leave residual predicted "
                "pressure at or above 90%; no safe low-risk action is verified."
            ),
        }

    return best
