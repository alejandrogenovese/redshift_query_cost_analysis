"""
Redshift Query Cost Analyzer — Flask API
"""
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from parser import parse_explain, OPERATION_COMPLEXITY, DISTRIBUTION_EFFICIENCY
from calibration import (
    CalibrationPoint, CalibrationModel, DEFAULT_MODEL,
    save_calibration_point, get_calibration_points, delete_calibration_point,
    clear_calibration, fit_model, get_latest_model, get_confidence_level,
    save_estimation, get_estimation_history,
)
from calculator import (
    ClusterConfig, analyze_explain, compare_queries, get_pricing_info,
)

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)


# ── Serve Frontend ───────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Pricing & Config ─────────────────────────────────────────────────

@app.route("/api/pricing", methods=["GET"])
def api_pricing():
    return jsonify(get_pricing_info())


@app.route("/api/reference-data", methods=["GET"])
def api_reference():
    return jsonify({
        "operation_types": list(OPERATION_COMPLEXITY.keys()),
        "distribution_strategies": list(DISTRIBUTION_EFFICIENCY.keys()),
        "operation_complexity": OPERATION_COMPLEXITY,
        "distribution_efficiency": DISTRIBUTION_EFFICIENCY,
    })


# ── Analysis ─────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json or {}
    explain_text = data.get("explain_text", "")
    if not explain_text.strip():
        return jsonify({"error": "explain_text is required"}), 400

    cluster = ClusterConfig(
        node_type=data.get("node_type", "ra3.4xlarge"),
        num_nodes=data.get("num_nodes", 2),
        reserved_type=data.get("reserved_type", "none"),
        avg_concurrency=data.get("avg_concurrency", 5),
        custom_hourly_rate=data.get("custom_hourly_rate"),
        billing_model=data.get("billing_model", "provisioned"),
        serverless_base_rpu=data.get("serverless_base_rpu", 8),
    )

    executions = data.get("executions_per_day", 1)
    result = analyze_explain(explain_text, cluster, executions)

    # Save to history
    save_estimation(
        explain_text[:500],
        result.estimated_seconds,
        result.total_cost,
        result.confidence,
        result.to_dict(),
    )

    # Parse plan for detailed view
    plan = parse_explain(explain_text)

    return jsonify({
        "estimate": result.to_dict(),
        "plan": plan.to_dict(),
        "cluster": cluster.to_dict(),
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.json or {}
    queries = data.get("queries", [])
    if not queries:
        return jsonify({"error": "queries array is required"}), 400

    cluster = ClusterConfig(
        node_type=data.get("node_type", "ra3.4xlarge"),
        num_nodes=data.get("num_nodes", 2),
        reserved_type=data.get("reserved_type", "none"),
        avg_concurrency=data.get("avg_concurrency", 5),
        billing_model=data.get("billing_model", "provisioned"),
    )

    results = compare_queries(queries, cluster)
    return jsonify({"comparisons": results})


@app.route("/api/parse", methods=["POST"])
def api_parse():
    data = request.json or {}
    explain_text = data.get("explain_text", "")
    if not explain_text.strip():
        return jsonify({"error": "explain_text is required"}), 400

    plan = parse_explain(explain_text)
    return jsonify({"plan": plan.to_dict()})


# ── Calibration ──────────────────────────────────────────────────────

@app.route("/api/calibration/points", methods=["GET"])
def api_get_calibration():
    points = get_calibration_points()
    return jsonify({"points": [p.to_dict() for p in points]})


@app.route("/api/calibration/add", methods=["POST"])
def api_add_calibration():
    data = request.json or {}
    required = ["explain_cost", "actual_time_seconds", "rows", "width",
                "operation_type", "num_nodes", "node_type"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing: {field}"}), 400

    point = CalibrationPoint(
        explain_cost=float(data["explain_cost"]),
        actual_time_seconds=float(data["actual_time_seconds"]),
        rows=int(data["rows"]),
        width=int(data["width"]),
        operation_type=data["operation_type"],
        distribution=data.get("distribution"),
        num_nodes=int(data["num_nodes"]),
        node_type=data["node_type"],
        query_label=data.get("query_label", ""),
    )

    point_id = save_calibration_point(point)
    return jsonify({"id": point_id, "point": point.to_dict()})


@app.route("/api/calibration/add-from-explain", methods=["POST"])
def api_add_from_explain():
    """Add calibration point by providing EXPLAIN text + actual time."""
    data = request.json or {}
    explain_text = data.get("explain_text", "")
    actual_time = data.get("actual_time_seconds")
    if not explain_text or actual_time is None:
        return jsonify({"error": "explain_text and actual_time_seconds required"}), 400

    plan = parse_explain(explain_text)
    if not plan.nodes:
        return jsonify({"error": "Could not parse EXPLAIN plan"}), 400

    point = CalibrationPoint(
        explain_cost=plan.root_cost,
        actual_time_seconds=float(actual_time),
        rows=plan.root_rows,
        width=plan.root_width,
        operation_type=plan.primary_operation or "Unknown",
        distribution=plan.primary_distribution,
        num_nodes=int(data.get("num_nodes", 2)),
        node_type=data.get("node_type", "ra3.4xlarge"),
        query_label=data.get("query_label", ""),
    )

    point_id = save_calibration_point(point)
    return jsonify({
        "id": point_id,
        "point": point.to_dict(),
        "parsed_plan": plan.to_dict(),
    })


@app.route("/api/calibration/delete/<int:point_id>", methods=["DELETE"])
def api_delete_calibration(point_id):
    delete_calibration_point(point_id)
    return jsonify({"deleted": point_id})


@app.route("/api/calibration/clear", methods=["POST"])
def api_clear_calibration():
    clear_calibration()
    return jsonify({"status": "cleared"})


@app.route("/api/calibration/fit", methods=["POST"])
def api_fit_model():
    points = get_calibration_points()
    if not points:
        return jsonify({"error": "No calibration points. Add benchmark data first."}), 400

    model = fit_model(points)
    confidence = get_confidence_level(model)

    return jsonify({
        "model": model.to_dict(),
        "confidence": confidence,
        "message": f"Modelo calibrado con {model.num_points} puntos (R²={model.r_squared:.3f})"
    })


@app.route("/api/calibration/model", methods=["GET"])
def api_get_model():
    model = get_latest_model()
    if model:
        return jsonify({
            "model": model.to_dict(),
            "confidence": get_confidence_level(model),
            "is_default": False,
        })
    return jsonify({
        "model": DEFAULT_MODEL.to_dict(),
        "confidence": "low",
        "is_default": True,
    })


# ── History ──────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def api_history():
    limit = request.args.get("limit", 50, type=int)
    history = get_estimation_history(limit)
    return jsonify({"history": history})


# ── Health ───────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def api_health():
    model = get_latest_model()
    points = get_calibration_points()
    return jsonify({
        "status": "ok",
        "calibration_points": len(points),
        "model_available": model is not None,
        "confidence": get_confidence_level(model) if model else "low",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
