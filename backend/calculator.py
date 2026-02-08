"""
Redshift Query Cost Calculator
Converts EXPLAIN plan data to monetary cost estimates using calibrated model.
"""
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict

from parser import ExplainPlan, parse_explain, get_parallelization_factor
from calibration import (
    CalibrationModel, DEFAULT_MODEL, get_latest_model,
    estimate_time, get_confidence_level
)


# ── Pricing Data ─────────────────────────────────────────────────────

NODE_PRICING = {
    # On-demand USD/hour per node
    "dc2.large":     0.25,
    "dc2.8xlarge":   4.80,
    "ra3.xlplus":    1.086,
    "ra3.4xlarge":   3.26,
    "ra3.16xlarge": 13.04,
    "ds2.xlarge":    0.85,
    "ds2.8xlarge":   6.80,
}

RESERVED_DISCOUNTS = {
    "none":                  0.00,
    "1yr_no_upfront":        0.20,
    "1yr_partial_upfront":   0.33,
    "1yr_all_upfront":       0.42,
    "3yr_no_upfront":        0.36,
    "3yr_partial_upfront":   0.53,
    "3yr_all_upfront":       0.63,
}

SERVERLESS_RPU_RATE = 0.375   # $/RPU-hour
SPECTRUM_RATE = 5.0            # $/TB scanned
RMS_RATE = 0.024               # $/GB-month


@dataclass
class ClusterConfig:
    node_type: str = "ra3.4xlarge"
    num_nodes: int = 2
    reserved_type: str = "none"
    avg_concurrency: int = 5       # average concurrent queries
    custom_hourly_rate: Optional[float] = None
    billing_model: str = "provisioned"   # "provisioned" or "serverless"
    serverless_base_rpu: int = 8

    @property
    def hourly_rate_per_node(self) -> float:
        if self.custom_hourly_rate:
            return self.custom_hourly_rate
        base = NODE_PRICING.get(self.node_type, 3.26)
        discount = RESERVED_DISCOUNTS.get(self.reserved_type, 0.0)
        return base * (1 - discount)

    @property
    def total_hourly_rate(self) -> float:
        return self.hourly_rate_per_node * self.num_nodes

    def to_dict(self):
        d = asdict(self)
        d["hourly_rate_per_node"] = self.hourly_rate_per_node
        d["total_hourly_rate"] = self.total_hourly_rate
        return d


@dataclass
class CostEstimate:
    # Time estimates
    estimated_seconds: float = 0.0
    estimated_minutes: float = 0.0
    estimated_hours: float = 0.0

    # Cost breakdown
    compute_cost: float = 0.0
    spectrum_cost: float = 0.0
    total_cost: float = 0.0

    # Formatted
    formatted_time: str = ""
    formatted_cost: str = ""
    formatted_compute: str = ""
    formatted_spectrum: str = ""

    # Plan analysis
    complexity_score: float = 1.0
    primary_operation: str = ""
    primary_distribution: Optional[str] = None
    parallelization_efficiency: float = 0.7
    data_scanned_gb: float = 0.0
    num_plan_nodes: int = 0

    # Confidence
    confidence: str = "low"
    confidence_detail: str = ""

    # Model info
    model_type: str = "default"
    model_r_squared: float = 0.0
    model_points: int = 0

    # Projections
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    yearly_cost: float = 0.0

    def to_dict(self):
        return asdict(self)


def _format_time(seconds: float) -> str:
    if seconds < 0.1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seg"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.2f} horas"


def _format_cost(usd: float) -> str:
    if usd < 0.01:
        return f"${usd:.6f}"
    elif usd < 1:
        return f"${usd:.4f}"
    elif usd < 100:
        return f"${usd:.2f}"
    else:
        return f"${usd:,.2f}"


def analyze_explain(explain_text: str, cluster: ClusterConfig,
                    executions_per_day: int = 1) -> CostEstimate:
    """
    Main analysis function.
    Takes EXPLAIN text + cluster config, returns full cost estimate.
    """
    # 1. Parse the EXPLAIN
    plan = parse_explain(explain_text)
    if not plan.nodes:
        return CostEstimate(confidence="none", confidence_detail="No se pudo parsear el EXPLAIN plan")

    # 2. Load calibration model
    model = get_latest_model() or DEFAULT_MODEL
    confidence = get_confidence_level(model)

    # 3. Estimate time
    est_seconds = estimate_time(
        model=model,
        cost=plan.root_cost,
        rows=plan.root_rows,
        width=plan.root_width,
        operation=plan.primary_operation,
        distribution=plan.primary_distribution,
    )

    # 4. Apply parallelization
    par_factor = get_parallelization_factor(plan, cluster.num_nodes)
    est_seconds *= par_factor

    # 5. Calculate monetary cost
    est_hours = est_seconds / 3600

    if cluster.billing_model == "serverless":
        # Serverless: RPU-hours
        rpu_hours = est_hours * cluster.serverless_base_rpu
        compute_cost = rpu_hours * SERVERLESS_RPU_RATE
    else:
        # Provisioned: proportional share of cluster
        # Query uses (est_time / total_time) of cluster, divided by concurrency
        compute_cost = est_hours * cluster.total_hourly_rate / max(cluster.avg_concurrency, 1)

    # 6. Spectrum cost (if applicable — estimate from data scanned)
    data_scanned_bytes = plan.estimated_data_scanned_bytes
    data_scanned_gb = data_scanned_bytes / (1024 ** 3)
    data_scanned_tb = data_scanned_gb / 1024
    spectrum_cost = data_scanned_tb * SPECTRUM_RATE if data_scanned_tb > 0 else 0.0

    total_cost = compute_cost + spectrum_cost

    # 7. Projections
    daily_cost = total_cost * executions_per_day
    monthly_cost = daily_cost * 30
    yearly_cost = daily_cost * 365

    # 8. Confidence detail
    if confidence == "high":
        detail = f"Modelo calibrado con {model.num_points} puntos (R²={model.r_squared:.2f})"
    elif confidence == "medium":
        detail = f"Modelo con calibración parcial ({model.num_points} puntos)"
    else:
        detail = "Modelo por defecto sin calibrar. Ejecute el wizard de calibración para mayor precisión."

    par_efficiency = 1.0
    if cluster.num_nodes > 1:
        par_efficiency = (1 / par_factor) / cluster.num_nodes

    return CostEstimate(
        estimated_seconds=est_seconds,
        estimated_minutes=est_seconds / 60,
        estimated_hours=est_hours,
        compute_cost=compute_cost,
        spectrum_cost=spectrum_cost,
        total_cost=total_cost,
        formatted_time=_format_time(est_seconds),
        formatted_cost=_format_cost(total_cost),
        formatted_compute=_format_cost(compute_cost),
        formatted_spectrum=_format_cost(spectrum_cost),
        complexity_score=plan.complexity_score,
        primary_operation=plan.primary_operation,
        primary_distribution=plan.primary_distribution,
        parallelization_efficiency=par_efficiency,
        data_scanned_gb=data_scanned_gb,
        num_plan_nodes=len(plan.nodes),
        confidence=confidence,
        confidence_detail=detail,
        model_type="calibrated" if model.num_points > 0 else "default",
        model_r_squared=model.r_squared,
        model_points=model.num_points,
        daily_cost=daily_cost,
        monthly_cost=monthly_cost,
        yearly_cost=yearly_cost,
    )


def compare_queries(explain_texts: list, cluster: ClusterConfig) -> list:
    """Compare multiple EXPLAIN plans and rank by cost."""
    results = []
    for i, text in enumerate(explain_texts):
        est = analyze_explain(text, cluster)
        results.append({
            "index": i,
            "query_label": f"Query {i + 1}",
            "estimate": est.to_dict(),
        })

    # Sort by total cost descending
    results.sort(key=lambda x: x["estimate"]["total_cost"], reverse=True)

    # Add ranking
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        if results[0]["estimate"]["total_cost"] > 0:
            r["relative_cost"] = (
                r["estimate"]["total_cost"] / results[0]["estimate"]["total_cost"]
            )
        else:
            r["relative_cost"] = 0

    return results


def get_pricing_info():
    """Return all pricing data for frontend."""
    return {
        "node_types": NODE_PRICING,
        "reserved_discounts": RESERVED_DISCOUNTS,
        "serverless_rpu_rate": SERVERLESS_RPU_RATE,
        "spectrum_rate": SPECTRUM_RATE,
        "rms_rate": RMS_RATE,
    }
