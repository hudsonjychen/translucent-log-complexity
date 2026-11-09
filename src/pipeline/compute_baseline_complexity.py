import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "process-complexity"))

import pandas as pd
from typing import List, Sequence
from pm4py.objects.log.obj import EventLog, Trace, Event
from tqdm.auto import tqdm
import Complexity
from .generate_translucent_logs import Log


CASE = 'case:concept:name'
ACTIVITY = 'concept:name'
TIMESTAMP = 'time:timestamp'

# Measures understood by Complexity.perform_measurements
DEFAULT_MEASURES = [
    "magnitude", "support", "variety", "level_of_detail", "time_granularity",
    "structure", "affinity", "trace_length", "distinct_traces",
    "deviation_from_random", "lempel-ziv", "pentland", "pentland_process",
]


def _to_pm4py_log(df: pd.DataFrame, case_id: str, activity: str, timestamp: str) -> EventLog:
    """Build a minimal pm4py EventLog in the shape Complexity.py expects
    (trace attribute 'concept:name' = case id, event keys 'concept:name' / 'time:timestamp')."""
    df = df[[case_id, activity, timestamp]].copy()
    df[timestamp] = pd.to_datetime(df[timestamp], utc=True)
    df = df.sort_values([case_id, timestamp], kind="stable")

    traces = []
    for cid, group in df.groupby(case_id, sort=False):
        trace = Trace(attributes={"concept:name": cid})
        for act, ts in zip(group[activity], group[timestamp]):
            trace.append(Event({"concept:name": act, "time:timestamp": ts.to_pydatetime()}))
        traces.append(trace)
    return EventLog(traces)


def _flatten(measurements: dict) -> dict:
    """Turn Complexity's display names into snake_case columns, expanding trace length."""
    row = {}
    for key, value in measurements.items():
        col = key.lower().replace("'s", "").replace("-", "_").replace(" ", "_")
        if isinstance(value, dict):
            for sub, v in value.items():
                row[f"{col}_{sub}"] = v
        else:
            row[col] = value
    return row


def compute_log_complexity(
    log: Log,
    case_id: str = CASE,
    activity: str = ACTIVITY,
    timestamp: str = TIMESTAMP,
    measures: Sequence[str] = DEFAULT_MEASURES,
    exp_k: float = 1.0,
) -> dict:
    """Compute Complexity.py measures and EPA entropies for a single Log."""
    pm4py_log = _to_pm4py_log(log["dataframe"], case_id, activity, timestamp)
    plain_log = Complexity.generate_log(pm4py_log)
    pa = Complexity.build_graph(plain_log)

    row = {"name": log["name"]}
    row.update(_flatten(Complexity.perform_measurements(
        list(measures), plain_log, pm4py_log, pa, quiet=True)))

    var_ent, var_ent_norm = Complexity.graph_complexity(pa)
    seq_ent, seq_ent_norm = Complexity.log_complexity(pa)
    seq_lin, seq_lin_norm = Complexity.log_complexity(pa, "linear")
    seq_exp, seq_exp_norm = Complexity.log_complexity(pa, "exp", exp_k)
    row.update({
        "variant_entropy": var_ent,
        "variant_entropy_norm": var_ent_norm,
        "sequence_entropy": seq_ent,
        "sequence_entropy_norm": seq_ent_norm,
        "sequence_entropy_linear": seq_lin,
        "sequence_entropy_linear_norm": seq_lin_norm,
        "sequence_entropy_exp": seq_exp,
        "sequence_entropy_exp_norm": seq_exp_norm,
    })
    return row


def compute_baseline_complexity(
    logs: List[Log],
    case_id: str = CASE,
    activity: str = ACTIVITY,
    timestamp: str = TIMESTAMP,
    measures: Sequence[str] = DEFAULT_MEASURES,
    exp_k: float = 1.0,
    zscore: bool = True,
) -> pd.DataFrame:
    """Compute baseline complexity measures (from external/process-complexity) for each Log.

    Returns one row per log. If ``zscore`` is True, a ``<col>_z`` column is added
    for every numeric measure, standardised across the given logs."""
    rows = [
        compute_log_complexity(log, case_id, activity, timestamp, measures, exp_k)
        for log in tqdm(logs, desc="Baseline Complexity")
    ]
    result = pd.DataFrame(rows)

    if zscore and len(result) > 1:
        metric_cols = [c for c in result.columns if c != "name"]
        for col in metric_cols:
            values = pd.to_numeric(result[col], errors="coerce")
            std = values.std()
            result[col + "_z"] = (values - values.mean()) / std if std else 0.0

    return result
