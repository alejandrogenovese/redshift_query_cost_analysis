"""
Calibration System for Redshift Cost Estimation
Builds regression model from user-provided benchmark data.
"""
import json
import math
import sqlite3
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timezone


DB_PATH = os.environ.get("CALIBRATION_DB", os.path.join(os.path.dirname(__file__), "calibration.db"))


@dataclass
class CalibrationPoint:
    """Single calibration data point from benchmark."""
    explain_cost: float
    actual_time_seconds: float
    rows: int
    width: int
    operation_type: str       # "Seq Scan", "Hash Join", etc.
    distribution: Optional[str]
    num_nodes: int
    node_type: str
    timestamp: str = ""
    query_label: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class CalibrationModel:
    """Fitted calibration model coefficients."""
    alpha: float = 0.0          # coefficient for log(cost)
    beta: float = 0.0           # coefficient for log(rows)
    gamma: float = 0.0          # coefficient for width
    intercept: float = 0.0
    # Per-operation adjustments
    operation_offsets: Dict[str, float] = None
    # Per-distribution adjustments
    distribution_offsets: Dict[str, float] = None
    # Metadata
    num_points: int = 0
    r_squared: float = 0.0
    mean_absolute_error: float = 0.0
    calibrated_at: str = ""
    cluster_node_type: str = ""
    cluster_num_nodes: int = 0

    def __post_init__(self):
        if self.operation_offsets is None:
            self.operation_offsets = {}
        if self.distribution_offsets is None:
            self.distribution_offsets = {}

    def to_dict(self):
        return asdict(self)


# ── Default Model (used when no calibration data available) ──────────

DEFAULT_MODEL = CalibrationModel(
    alpha=0.45,         # log(cost) coefficient
    beta=0.05,          # log(rows) coefficient
    gamma=0.0001,       # width coefficient
    intercept=-3.0,
    operation_offsets={
        "Seq Scan": 0.0,
        "Hash Join": 0.3,
        "Merge Join": 0.1,
        "Nested Loop": 0.8,
        "Sort": 0.2,
        "HashAggregate": 0.15,
        "GroupAggregate": 0.1,
        "WindowAgg": 0.4,
    },
    distribution_offsets={
        "DS_DIST_NONE": 0.0,
        "DS_DIST_ALL_NONE": 0.05,
        "DS_BCAST_INNER": 0.2,
        "DS_DIST_OUTER": 0.4,
        "DS_DIST_BOTH": 0.6,
    },
    num_points=0,
    r_squared=0.0,
    calibrated_at="default",
    cluster_node_type="generic",
    cluster_num_nodes=2,
)


# ── Database Functions ───────────────────────────────────────────────

def _get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE IF NOT EXISTS calibration_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            explain_cost REAL NOT NULL,
            actual_time_seconds REAL NOT NULL,
            rows INTEGER NOT NULL,
            width INTEGER NOT NULL,
            operation_type TEXT NOT NULL,
            distribution TEXT,
            num_nodes INTEGER NOT NULL,
            node_type TEXT NOT NULL,
            query_label TEXT DEFAULT '',
            timestamp TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS calibration_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS estimation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            explain_input TEXT,
            estimated_time REAL,
            estimated_cost REAL,
            confidence TEXT,
            details_json TEXT,
            created_at TEXT NOT NULL
        )
    """)
    db.commit()
    return db


def save_calibration_point(point: CalibrationPoint) -> int:
    db = _get_db()
    point.timestamp = datetime.now(timezone.utc).isoformat()
    cur = db.execute("""
        INSERT INTO calibration_points
        (explain_cost, actual_time_seconds, rows, width, operation_type,
         distribution, num_nodes, node_type, query_label, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        point.explain_cost, point.actual_time_seconds, point.rows, point.width,
        point.operation_type, point.distribution, point.num_nodes, point.node_type,
        point.query_label, point.timestamp
    ))
    db.commit()
    return cur.lastrowid


def get_calibration_points() -> List[CalibrationPoint]:
    db = _get_db()
    rows = db.execute("SELECT * FROM calibration_points ORDER BY timestamp DESC").fetchall()
    points = []
    for r in rows:
        points.append(CalibrationPoint(
            explain_cost=r["explain_cost"],
            actual_time_seconds=r["actual_time_seconds"],
            rows=r["rows"],
            width=r["width"],
            operation_type=r["operation_type"],
            distribution=r["distribution"],
            num_nodes=r["num_nodes"],
            node_type=r["node_type"],
            query_label=r["query_label"],
            timestamp=r["timestamp"],
        ))
    return points


def delete_calibration_point(point_id: int):
    db = _get_db()
    db.execute("DELETE FROM calibration_points WHERE id = ?", (point_id,))
    db.commit()


def clear_calibration():
    db = _get_db()
    db.execute("DELETE FROM calibration_points")
    db.execute("DELETE FROM calibration_models")
    db.commit()


def save_model(model: CalibrationModel):
    db = _get_db()
    db.execute("""
        INSERT INTO calibration_models (model_json, created_at)
        VALUES (?, ?)
    """, (json.dumps(model.to_dict()), datetime.now(timezone.utc).isoformat()))
    db.commit()


def get_latest_model() -> Optional[CalibrationModel]:
    db = _get_db()
    row = db.execute(
        "SELECT model_json FROM calibration_models ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row:
        d = json.loads(row["model_json"])
        return CalibrationModel(**d)
    return None


def save_estimation(explain_input, est_time, est_cost, confidence, details):
    db = _get_db()
    db.execute("""
        INSERT INTO estimation_history
        (explain_input, estimated_time, estimated_cost, confidence, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (explain_input, est_time, est_cost, confidence,
          json.dumps(details), datetime.now(timezone.utc).isoformat()))
    db.commit()


def get_estimation_history(limit=50):
    db = _get_db()
    rows = db.execute(
        "SELECT * FROM estimation_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── Regression Fitting ───────────────────────────────────────────────

def fit_model(points: List[CalibrationPoint]) -> CalibrationModel:
    """
    Fit a calibration model using least-squares regression.
    Model: log(time) = α·log(cost) + β·log(rows) + γ·width + δ[op] + ε[dist] + intercept
    
    Uses simple normal equations (no numpy dependency).
    """
    n = len(points)
    if n < 3:
        # Not enough points; return default with simple scaling
        if n >= 1:
            # Use average ratio as simple calibration
            ratios = []
            for p in points:
                if p.explain_cost > 0 and p.actual_time_seconds > 0:
                    ratios.append(math.log(p.actual_time_seconds) / max(math.log(p.explain_cost), 0.001))
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                model = CalibrationModel(
                    alpha=avg_ratio,
                    beta=0.0,
                    gamma=0.0,
                    intercept=0.0,
                    num_points=n,
                    r_squared=0.0,
                    calibrated_at=datetime.now(timezone.utc).isoformat(),
                )
                save_model(model)
                return model
        return DEFAULT_MODEL

    # Collect unique operations and distributions for one-hot encoding
    all_ops = list(set(p.operation_type for p in points))
    all_dists = list(set(p.distribution for p in points if p.distribution))

    # Build feature matrix and target
    # Features: log(cost), log(rows), width, [op one-hot], [dist one-hot]
    num_base_features = 3  # log_cost, log_rows, width
    num_features = num_base_features + len(all_ops) + len(all_dists) + 1  # +1 for intercept

    X = []
    y = []
    for p in points:
        if p.explain_cost <= 0 or p.actual_time_seconds <= 0:
            continue
        row = [
            math.log(max(p.explain_cost, 1.0)),
            math.log(max(p.rows, 1)),
            p.width / 1000.0,  # scale width
        ]
        # One-hot for operations
        for op in all_ops:
            row.append(1.0 if p.operation_type == op else 0.0)
        # One-hot for distributions
        for dist in all_dists:
            row.append(1.0 if p.distribution == dist else 0.0)
        # Intercept
        row.append(1.0)

        X.append(row)
        y.append(math.log(p.actual_time_seconds))

    if len(X) < 3:
        return DEFAULT_MODEL

    # Solve using normal equations: β = (X'X)^-1 X'y
    # Simple implementation without numpy
    m = len(X)
    k = len(X[0])

    # Compute X'X
    XtX = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            s = 0.0
            for r in range(m):
                s += X[r][i] * X[r][j]
            XtX[i][j] = s

    # Compute X'y
    Xty = [0.0] * k
    for i in range(k):
        s = 0.0
        for r in range(m):
            s += X[r][i] * y[r]
        Xty[i] = s

    # Solve via Gaussian elimination with partial pivoting
    coeffs = _solve_linear_system(XtX, Xty)

    if coeffs is None:
        # Fallback to simple ratio model
        ratios = []
        for p in points:
            if p.explain_cost > 0 and p.actual_time_seconds > 0:
                ratios.append(p.actual_time_seconds / math.log(max(p.explain_cost, 2.0)))
        avg = sum(ratios) / len(ratios) if ratios else 1.0
        model = CalibrationModel(
            alpha=1.0, beta=0.0, gamma=0.0, intercept=math.log(max(avg, 0.001)),
            num_points=n,
            calibrated_at=datetime.now(timezone.utc).isoformat(),
        )
        save_model(model)
        return model

    # Extract coefficients
    alpha = coeffs[0]
    beta = coeffs[1]
    gamma = coeffs[2]

    op_offsets = {}
    for i, op in enumerate(all_ops):
        op_offsets[op] = coeffs[num_base_features + i]

    dist_offsets = {}
    for i, dist in enumerate(all_dists):
        dist_offsets[dist] = coeffs[num_base_features + len(all_ops) + i]

    intercept = coeffs[-1]

    # Compute R²
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = 0.0
    abs_errors = []
    for r in range(m):
        y_pred = sum(X[r][j] * coeffs[j] for j in range(k))
        ss_res += (y[r] - y_pred) ** 2
        abs_errors.append(abs(math.exp(y[r]) - math.exp(y_pred)))

    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

    model = CalibrationModel(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        intercept=intercept,
        operation_offsets=op_offsets,
        distribution_offsets=dist_offsets,
        num_points=n,
        r_squared=max(0, min(1, r_squared)),
        mean_absolute_error=mae,
        calibrated_at=datetime.now(timezone.utc).isoformat(),
        cluster_node_type=points[0].node_type if points else "unknown",
        cluster_num_nodes=points[0].num_nodes if points else 2,
    )
    save_model(model)
    return model


def _solve_linear_system(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row
        if max_val < 1e-12:
            # Add regularization
            M[col][col] += 1e-6
        M[col], M[max_row] = M[max_row], M[col]

        # Eliminate
        pivot = M[col][col]
        if abs(pivot) < 1e-15:
            continue
        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-15:
            x[i] = 0.0
            continue
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


# ── Estimation ───────────────────────────────────────────────────────

def estimate_time(model: CalibrationModel, cost: float, rows: int, width: int,
                  operation: str = "", distribution: str = None) -> float:
    """
    Estimate execution time in seconds using calibration model.
    Returns: estimated seconds
    """
    if cost <= 0:
        return 0.0

    log_cost = math.log(max(cost, 1.0))
    log_rows = math.log(max(rows, 1))
    scaled_width = width / 1000.0

    log_time = (
        model.alpha * log_cost
        + model.beta * log_rows
        + model.gamma * scaled_width
        + model.intercept
    )

    # Operation offset
    if operation:
        base_op = operation.split()[0]
        offset = model.operation_offsets.get(
            operation,
            model.operation_offsets.get(base_op, 0.0)
        )
        log_time += offset

    # Distribution offset
    if distribution and distribution in model.distribution_offsets:
        log_time += model.distribution_offsets[distribution]

    # Convert from log-space
    estimated_seconds = math.exp(log_time)

    # Sanity bounds
    estimated_seconds = max(0.001, min(estimated_seconds, 86400 * 7))  # 1ms to 7 days

    return estimated_seconds


def get_confidence_level(model: CalibrationModel) -> str:
    """Determine confidence level of the model."""
    if model.calibrated_at == "default" or model.num_points == 0:
        return "low"
    elif model.num_points < 5:
        return "medium"
    elif model.r_squared > 0.7:
        return "high"
    elif model.r_squared > 0.4:
        return "medium"
    else:
        return "low"
