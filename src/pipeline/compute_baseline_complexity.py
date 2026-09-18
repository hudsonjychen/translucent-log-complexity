import pandas as pd
from typing import List
from ..baseline_complexity_measurements.baseline_complexity import variety, structure, affinity
from generate_translucent_logs import Log


CASE = 'case:concept:name'
ACTIVITY = 'concept:name'
TIMESTAMP = 'time:timestamp'

def _extract_traces(df, case_id, activity, timestamp):
    df = df.sort_values(by=[case_id, timestamp])
    return [tuple(g) for _, g in df.groupby(case_id, sort=False)[activity]]

def compute_baseline_complexity(
    logs: List[Log],
    case_id: str = CASE,
    activity: str = ACTIVITY,
    timestamp: str = TIMESTAMP,
) -> pd.DataFrame:
    rows = []
    for log in logs:
        df = log["dataframe"]
        traces = _extract_traces(df, case_id, activity, timestamp)
        rows.append({
            "name": log["name"],
            "variety": variety(traces),
            "structure": structure(traces),
            "affinity": affinity(traces),
        })

    result = pd.DataFrame(rows, columns=["name", "variety", "structure", "affinity"])
    for col in ["variety", "structure", "affinity"]:
        result[col + "_z"] = (result[col] - result[col].mean()) / result[col].std()

    return result