"""Tests for Redshift Query Cost Analyzer."""
import os
import sys
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
os.environ["CALIBRATION_DB"] = ":memory:"

from parser import parse_explain, parse_single_cost, get_parallelization_factor, ExplainPlan
from calibration import (
    CalibrationPoint, CalibrationModel, DEFAULT_MODEL,
    fit_model, estimate_time, get_confidence_level, _solve_linear_system,
)
from calculator import ClusterConfig, analyze_explain, compare_queries, get_pricing_info


# ── Parser Tests ─────────────────────────────────────────────────────

class TestParser:
    def test_single_cost_line(self):
        r = parse_single_cost("(cost=0.00..883328.16 rows=88332816 width=108)")
        assert r is not None
        assert r["total_cost"] == 883328.16
        assert r["rows"] == 88332816
        assert r["width"] == 108
        assert r["startup_cost"] == 0.0

    def test_multiline_explain(self):
        explain = """QUERY PLAN
---
XN Hash Join DS_BCAST_INNER  (cost=175734.03..1652544172278.50 rows=6938483 width=179)
  Hash Cond: ("outer".id = "inner".id)
  ->  XN Seq Scan on orders  (cost=0.00..69384.83 rows=6938483 width=108)
  ->  XN Hash  (cost=1757.34..1757.34 rows=175734 width=71)
        ->  XN Seq Scan on customers  (cost=0.00..1757.34 rows=175734 width=71)"""
        plan = parse_explain(explain)
        assert len(plan.nodes) >= 3
        assert plan.root_cost == pytest.approx(1652544172278.50, rel=1e-6)
        assert plan.has_hash_join or plan.has_seq_scan
        assert plan.has_broadcast

    def test_simple_seq_scan(self):
        explain = "XN Seq Scan on event (cost=0.00..87.98 rows=8798 width=17)"
        plan = parse_explain(explain)
        assert len(plan.nodes) == 1
        assert plan.root_cost == 87.98
        assert plan.root_rows == 8798
        assert plan.has_seq_scan

    def test_distribution_detection(self):
        explain = """XN Hash Join DS_DIST_BOTH  (cost=100.00..5000.00 rows=1000 width=50)
  ->  XN Seq Scan on t1  (cost=0.00..50.00 rows=500 width=25)
  ->  XN Seq Scan on t2  (cost=0.00..50.00 rows=500 width=25)"""
        plan = parse_explain(explain)
        assert plan.primary_distribution == "DS_DIST_BOTH"
        assert plan.has_redistribute

    def test_parallelization_factor(self):
        plan = ExplainPlan()
        plan.primary_distribution = "DS_DIST_NONE"
        f = get_parallelization_factor(plan, 4)
        assert f < 1.0
        assert f > 0  # Should be 1 / (1 + 3 * 0.90)

        plan2 = ExplainPlan()
        plan2.primary_distribution = "DS_DIST_BOTH"
        f2 = get_parallelization_factor(plan2, 4)
        assert f2 > f  # Worse distribution = higher factor (slower)

    def test_complexity_score(self):
        simple = parse_explain("XN Seq Scan on t (cost=0.00..100.00 rows=1000 width=10)")
        complex_q = parse_explain("""XN Hash Join DS_DIST_OUTER  (cost=100.00..5000.00 rows=1000 width=50)
  ->  XN Seq Scan on t1  (cost=0.00..50.00 rows=500 width=25)
  ->  XN Seq Scan on t2  (cost=0.00..50.00 rows=500 width=25)""")
        assert complex_q.complexity_score > simple.complexity_score

    def test_empty_input(self):
        plan = parse_explain("")
        assert len(plan.nodes) == 0

    def test_hash_aggregate(self):
        explain = """XN HashAggregate  (cost=131.97..133.41 rows=576 width=17)
  ->  XN Seq Scan on event  (cost=0.00..87.98 rows=8798 width=17)"""
        plan = parse_explain(explain)
        assert plan.has_aggregate
        assert len(plan.nodes) == 2


# ── Calibration Tests ────────────────────────────────────────────────

class TestCalibration:
    def test_default_model_estimate(self):
        t = estimate_time(DEFAULT_MODEL, cost=1000, rows=10000, width=100)
        assert t > 0
        assert t < 86400 * 7  # Less than 7 days

    def test_higher_cost_higher_time(self):
        t1 = estimate_time(DEFAULT_MODEL, cost=100, rows=1000, width=50)
        t2 = estimate_time(DEFAULT_MODEL, cost=100000, rows=1000000, width=50)
        assert t2 > t1

    def test_operation_offsets(self):
        t_scan = estimate_time(DEFAULT_MODEL, cost=1000, rows=10000, width=50, operation="Seq Scan")
        t_nested = estimate_time(DEFAULT_MODEL, cost=1000, rows=10000, width=50, operation="Nested Loop")
        assert t_nested > t_scan  # Nested loop should be more expensive

    def test_fit_model_minimal(self):
        points = [
            CalibrationPoint(100, 0.5, 1000, 50, "Seq Scan", None, 2, "ra3.4xlarge"),
            CalibrationPoint(10000, 5.0, 100000, 100, "Hash Join", "DS_BCAST_INNER", 2, "ra3.4xlarge"),
            CalibrationPoint(1000000, 30.0, 1000000, 150, "Sort", None, 2, "ra3.4xlarge"),
        ]
        model = fit_model(points)
        assert model.num_points == 3
        assert model.alpha != 0  # Should have fitted

    def test_fit_model_with_more_points(self):
        points = [
            CalibrationPoint(50, 0.1, 500, 20, "Seq Scan", "DS_DIST_NONE", 2, "ra3.4xlarge"),
            CalibrationPoint(500, 0.8, 5000, 50, "Seq Scan", "DS_DIST_NONE", 2, "ra3.4xlarge"),
            CalibrationPoint(5000, 3.5, 50000, 80, "Hash Join", "DS_BCAST_INNER", 2, "ra3.4xlarge"),
            CalibrationPoint(50000, 15.0, 500000, 100, "Hash Join", "DS_DIST_OUTER", 2, "ra3.4xlarge"),
            CalibrationPoint(500000, 45.0, 5000000, 120, "Sort", None, 2, "ra3.4xlarge"),
        ]
        model = fit_model(points)
        assert model.num_points == 5
        assert model.r_squared > 0  # Should explain some variance

    def test_confidence_levels(self):
        assert get_confidence_level(DEFAULT_MODEL) == "low"
        m = CalibrationModel(num_points=3, r_squared=0.5, calibrated_at="2025-01-01")
        assert get_confidence_level(m) == "medium"
        m2 = CalibrationModel(num_points=10, r_squared=0.85, calibrated_at="2025-01-01")
        assert get_confidence_level(m2) == "high"

    def test_solve_linear_system(self):
        # Simple 2x2: 2x + y = 5, x + 3y = 10
        A = [[2, 1], [1, 3]]
        b = [5, 10]
        x = _solve_linear_system(A, b)
        assert abs(x[0] - 1.0) < 1e-6
        assert abs(x[1] - 3.0) < 1e-6


# ── Calculator Tests ─────────────────────────────────────────────────

class TestCalculator:
    def test_basic_analysis(self):
        explain = "XN Seq Scan on orders (cost=0.00..69384.83 rows=6938483 width=108)"
        cluster = ClusterConfig(node_type="ra3.4xlarge", num_nodes=2)
        result = analyze_explain(explain, cluster)
        assert result.estimated_seconds > 0
        assert result.total_cost > 0
        assert result.formatted_cost.startswith("$")
        assert result.confidence in ("low", "medium", "high")

    def test_serverless_billing(self):
        explain = "XN Seq Scan on orders (cost=0.00..69384.83 rows=6938483 width=108)"
        prov = ClusterConfig(billing_model="provisioned", num_nodes=2)
        svls = ClusterConfig(billing_model="serverless", serverless_base_rpu=8)
        r_prov = analyze_explain(explain, prov)
        r_svls = analyze_explain(explain, svls)
        assert r_prov.total_cost > 0
        assert r_svls.total_cost > 0
        # Both should give some cost
        assert r_prov.compute_cost != r_svls.compute_cost  # Different models

    def test_concurrency_reduces_cost(self):
        explain = "XN Seq Scan on big_table (cost=0.00..1000000.00 rows=100000000 width=200)"
        c1 = ClusterConfig(avg_concurrency=1, num_nodes=2)
        c5 = ClusterConfig(avg_concurrency=5, num_nodes=2)
        r1 = analyze_explain(explain, c1)
        r5 = analyze_explain(explain, c5)
        assert r1.compute_cost > r5.compute_cost  # Higher concurrency = lower per-query cost

    def test_reserved_discount(self):
        explain = "XN Seq Scan on t (cost=0.00..10000.00 rows=100000 width=50)"
        on_demand = ClusterConfig(reserved_type="none")
        reserved = ClusterConfig(reserved_type="3yr_all_upfront")
        r_od = analyze_explain(explain, on_demand)
        r_ri = analyze_explain(explain, reserved)
        assert r_ri.compute_cost < r_od.compute_cost

    def test_compare_queries(self):
        q1 = "XN Seq Scan on small_table (cost=0.00..100.00 rows=1000 width=20)"
        q2 = "XN Seq Scan on big_table (cost=0.00..1000000.00 rows=100000000 width=200)"
        cluster = ClusterConfig()
        results = compare_queries([q1, q2], cluster)
        assert len(results) == 2
        assert results[0]["rank"] == 1  # Most expensive first
        assert results[0]["estimate"]["total_cost"] >= results[1]["estimate"]["total_cost"]

    def test_pricing_info(self):
        info = get_pricing_info()
        assert "node_types" in info
        assert "ra3.4xlarge" in info["node_types"]
        assert info["serverless_rpu_rate"] == 0.375

    def test_projections(self):
        explain = "XN Seq Scan on t (cost=0.00..10000.00 rows=100000 width=50)"
        cluster = ClusterConfig()
        result = analyze_explain(explain, cluster, executions_per_day=10)
        assert result.daily_cost == pytest.approx(result.total_cost * 10, rel=1e-6)
        assert result.monthly_cost == pytest.approx(result.daily_cost * 30, rel=1e-6)

    def test_empty_explain(self):
        result = analyze_explain("", ClusterConfig())
        assert result.confidence == "none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
