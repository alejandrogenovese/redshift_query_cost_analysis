"""
Enhanced Redshift EXPLAIN Parser
Extracts: cost units, rows, width, operation type, join type, distribution strategy
"""
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ExplainNode:
    """Single node from an EXPLAIN plan tree."""
    operation: str              # e.g. "Seq Scan", "Hash Join", "Sort"
    table_name: Optional[str]   # e.g. "acct", "event"
    startup_cost: float
    total_cost: float
    rows: int
    width: int
    depth: int                  # indentation level in plan tree
    distribution: Optional[str] = None   # DS_DIST_NONE, DS_BCAST_INNER, etc.
    join_type: Optional[str] = None      # Hash Join, Merge Join, Nested Loop
    filter_condition: Optional[str] = None
    sort_key: Optional[str] = None
    hash_condition: Optional[str] = None
    merge_condition: Optional[str] = None
    raw_line: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ExplainPlan:
    """Complete parsed EXPLAIN plan."""
    nodes: List[ExplainNode] = field(default_factory=list)
    root_cost: float = 0.0
    root_rows: int = 0
    root_width: int = 0
    primary_operation: str = ""
    primary_distribution: Optional[str] = None
    has_seq_scan: bool = False
    has_nested_loop: bool = False
    has_hash_join: bool = False
    has_merge_join: bool = False
    has_sort: bool = False
    has_aggregate: bool = False
    has_broadcast: bool = False
    has_redistribute: bool = False
    estimated_data_scanned_bytes: int = 0
    complexity_score: float = 1.0

    def to_dict(self):
        d = asdict(self)
        d["nodes"] = [n.to_dict() for n in self.nodes]
        return d


# Patterns
COST_PATTERN = re.compile(
    r'\(cost=([0-9.]+)\.\.([0-9.]+)\s+rows=([0-9]+)\s+width=([0-9]+)\)'
)
OPERATION_PATTERN = re.compile(
    r'^(\s*)->\s*XN\s+(.+?)(?:\s+\(cost=)', re.MULTILINE
)
ROOT_OPERATION_PATTERN = re.compile(
    r'^(\s*)XN\s+(.+?)(?:\s+\(cost=)', re.MULTILINE
)
DISTRIBUTION_PATTERN = re.compile(
    r'(DS_DIST_NONE|DS_DIST_ALL_NONE|DS_DIST_ALL_INNER|DS_BCAST_INNER|'
    r'DS_DIST_OUTER|DS_DIST_INNER|DS_DIST_BOTH)'
)
TABLE_PATTERN = re.compile(r'on\s+(\w+)(?:\s|$)')
FILTER_PATTERN = re.compile(r'Filter:\s+(.+)')
SORT_KEY_PATTERN = re.compile(r'Sort Key:\s+(.+)')
HASH_COND_PATTERN = re.compile(r'Hash Cond:\s+(.+)')
MERGE_COND_PATTERN = re.compile(r'Merge Cond:\s+(.+)')

# Operation classification
SCAN_OPS = {"Seq Scan", "Index Scan", "Bitmap Heap Scan", "Bitmap Index Scan",
            "RLS SecureScan", "LF SecureScan", "Subquery Scan"}
JOIN_OPS = {"Hash Join", "Merge Join", "Nested Loop"}
AGG_OPS = {"HashAggregate", "GroupAggregate", "Aggregate", "WindowAgg"}
SORT_OPS = {"Sort", "Merge"}
NETWORK_OPS = {"Network"}
LIMIT_OPS = {"Limit"}

# Complexity weights by operation type
OPERATION_COMPLEXITY = {
    "Seq Scan": 1.0,
    "Index Scan": 0.6,
    "Bitmap Heap Scan": 0.8,
    "Hash Join": 1.4,
    "Merge Join": 1.1,
    "Nested Loop": 2.0,
    "Sort": 1.3,
    "HashAggregate": 1.2,
    "GroupAggregate": 1.1,
    "WindowAgg": 1.5,
    "Unique": 1.0,
    "Limit": 0.5,
    "Append": 1.0,
    "Result": 0.5,
    "Subquery Scan": 1.0,
    "Network": 1.2,
    "Merge": 1.1,
}

# Distribution efficiency factors
DISTRIBUTION_EFFICIENCY = {
    "DS_DIST_NONE": 0.90,
    "DS_DIST_ALL_NONE": 0.85,
    "DS_DIST_ALL_INNER": 0.80,
    "DS_BCAST_INNER": 0.65,
    "DS_DIST_OUTER": 0.55,
    "DS_DIST_INNER": 0.55,
    "DS_DIST_BOTH": 0.45,
}


def _parse_operation_name(op_text: str) -> tuple:
    """Extract clean operation name and optional distribution/table."""
    op_text = op_text.strip()
    distribution = None
    table_name = None

    dist_match = DISTRIBUTION_PATTERN.search(op_text)
    if dist_match:
        distribution = dist_match.group(1)

    table_match = TABLE_PATTERN.search(op_text)
    if table_match:
        table_name = table_match.group(1)

    # Clean operation name
    clean = op_text
    clean = DISTRIBUTION_PATTERN.sub('', clean)
    clean = TABLE_PATTERN.sub('', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Remove alias like "e" or "atc"
    clean = re.sub(r'\s+\w{1,3}$', '', clean).strip()

    return clean, distribution, table_name


def _classify_join(op_name: str) -> Optional[str]:
    """Classify join type from operation name."""
    lower = op_name.lower()
    if "hash join" in lower or "hash left join" in lower or "hash right join" in lower:
        return "Hash Join"
    elif "merge join" in lower:
        return "Merge Join"
    elif "nested loop" in lower:
        return "Nested Loop"
    return None


def parse_explain(explain_text: str) -> ExplainPlan:
    """Parse multi-line EXPLAIN output into structured ExplainPlan."""
    plan = ExplainPlan()
    lines = explain_text.strip().split('\n')

    # Remove header line if present
    lines = [l for l in lines if l.strip() and l.strip() != "QUERY PLAN" and not l.strip().startswith("---")]

    pending_filter = None
    pending_sort_key = None
    pending_hash_cond = None
    pending_merge_cond = None

    for line in lines:
        # Check for supplementary info lines
        filter_match = FILTER_PATTERN.search(line)
        if filter_match and plan.nodes:
            plan.nodes[-1].filter_condition = filter_match.group(1).strip()
            continue

        sort_match = SORT_KEY_PATTERN.search(line)
        if sort_match and plan.nodes:
            plan.nodes[-1].sort_key = sort_match.group(1).strip()
            continue

        hash_match = HASH_COND_PATTERN.search(line)
        if hash_match and plan.nodes:
            plan.nodes[-1].hash_condition = hash_match.group(1).strip()
            continue

        merge_match = MERGE_COND_PATTERN.search(line)
        if merge_match and plan.nodes:
            plan.nodes[-1].merge_condition = merge_match.group(1).strip()
            continue

        # Parse cost line
        cost_match = COST_PATTERN.search(line)
        if not cost_match:
            continue

        startup_cost = float(cost_match.group(1))
        total_cost = float(cost_match.group(2))
        rows = int(cost_match.group(3))
        width = int(cost_match.group(4))

        # Parse operation
        op_match = OPERATION_PATTERN.search(line)
        if not op_match:
            op_match = ROOT_OPERATION_PATTERN.search(line)

        if op_match:
            indent = len(op_match.group(1))
            depth = indent // 2  # typically 2-space or 3-space indent
            op_text = op_match.group(2)
        else:
            depth = 0
            # Try to extract operation from before (cost=
            before_cost = line.split('(cost=')[0].strip()
            before_cost = re.sub(r'^->\s*', '', before_cost).strip()
            before_cost = re.sub(r'^XN\s+', '', before_cost).strip()
            op_text = before_cost if before_cost else "Unknown"

        op_name, distribution, table_name = _parse_operation_name(op_text)
        join_type = _classify_join(op_text)

        node = ExplainNode(
            operation=op_name,
            table_name=table_name,
            startup_cost=startup_cost,
            total_cost=total_cost,
            rows=rows,
            width=width,
            depth=depth,
            distribution=distribution,
            join_type=join_type,
            raw_line=line.strip()
        )
        plan.nodes.append(node)

    if not plan.nodes:
        return plan

    # Set root node info
    root = plan.nodes[0]
    plan.root_cost = root.total_cost
    plan.root_rows = root.rows
    plan.root_width = root.width
    plan.primary_operation = root.operation

    # Analyze plan characteristics
    for node in plan.nodes:
        op = node.operation.lower()
        if "seq scan" in op:
            plan.has_seq_scan = True
            # Estimate data scanned: rows × width for scans
            plan.estimated_data_scanned_bytes += node.rows * node.width
        if "nested loop" in op:
            plan.has_nested_loop = True
        if "hash join" in op or "hash left join" in op:
            plan.has_hash_join = True
        if "merge join" in op:
            plan.has_merge_join = True
        if "sort" in op:
            plan.has_sort = True
        if any(a in op for a in ["aggregate", "hashaggregate", "groupaggregate", "windowagg"]):
            plan.has_aggregate = True

        if node.distribution:
            if "BCAST" in node.distribution:
                plan.has_broadcast = True
            if "DIST_OUTER" in node.distribution or "DIST_BOTH" in node.distribution or "DIST_INNER" in node.distribution:
                plan.has_redistribute = True
            if plan.primary_distribution is None:
                plan.primary_distribution = node.distribution

    # Calculate complexity score
    max_complexity = 1.0
    for node in plan.nodes:
        base_op = node.operation.split()[0] if node.operation else ""
        full_op = node.operation
        c = OPERATION_COMPLEXITY.get(full_op, OPERATION_COMPLEXITY.get(base_op, 1.0))
        if c > max_complexity:
            max_complexity = c

    # Adjust for distribution overhead
    dist_penalty = 1.0
    if plan.has_redistribute:
        dist_penalty = 1.3
    elif plan.has_broadcast:
        dist_penalty = 1.15

    plan.complexity_score = max_complexity * dist_penalty

    return plan


def parse_single_cost(cost_text: str) -> Optional[dict]:
    """Parse a single cost line like (cost=0.00..883328.16 rows=88332816 width=108)."""
    match = COST_PATTERN.search(cost_text)
    if match:
        return {
            "startup_cost": float(match.group(1)),
            "total_cost": float(match.group(2)),
            "rows": int(match.group(3)),
            "width": int(match.group(4))
        }
    return None


def get_parallelization_factor(plan: ExplainPlan, num_nodes: int) -> float:
    """Get parallelization efficiency based on distribution strategy."""
    if num_nodes <= 1:
        return 1.0

    dist = plan.primary_distribution
    if dist and dist in DISTRIBUTION_EFFICIENCY:
        efficiency = DISTRIBUTION_EFFICIENCY[dist]
    elif plan.has_redistribute:
        efficiency = 0.50
    elif plan.has_broadcast:
        efficiency = 0.65
    else:
        efficiency = 0.70  # default

    return 1 / (1 + (num_nodes - 1) * efficiency)
