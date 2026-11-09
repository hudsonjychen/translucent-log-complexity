"""
Count the activities of every translucent log and save them as a CSV (one row per log):
    num_activities_executed   |A_L^ex|  activities that occur as events
    num_activities            |A_L|     activities that are executed or appear in an enabled set
    num_activities_enabled_only         activities that are enabled at some point but never executed

The enabled sets are parsed like in src/complexity_measurements (split on ","), so |A_L| is the
same activity set the complexity metrics use (e.g. for avg-norm).

Usage (from the repo root):
    venv\\Scripts\\python.exe scripts\\evaluation\\get_activity_number.py
    venv\\Scripts\\python.exe scripts\\evaluation\\get_activity_number.py --input results\\translucent_simulated\\translucent_simulated_logs.pkl --output results\\evaluation\\simulated\\activity_number.csv
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ACTIVITY = "concept:name"
ENABLED_ACTIVITIES = "enabled_activities"
SEP = ","


def get_activity_number(logs: list[dict], activity_key: str = ACTIVITY,
                        enabled_activities_key: str = ENABLED_ACTIVITIES) -> pd.DataFrame:
    rows = []
    for log in logs:
        df = log["dataframe"]
        executed = {a for a in df[activity_key].astype(str).str.strip() if a and a != "nan"}
        enabled = {
            a.strip()
            for s in df[enabled_activities_key].dropna().astype(str)
            for a in s.split(SEP) if a.strip()
        }
        rows.append({
            "name": log["name"],
            "num_activities_executed": len(executed),
            "num_activities": len(executed | enabled),
            "num_activities_enabled_only": len(enabled - executed),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "results" / "translucent_real_life" / "translucent_real_life_logs.pkl")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation" / "real_life" / "activity_number.csv")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        logs = pickle.load(f)

    # a translucent log is empty when no test case fitted the model; there is nothing to count
    empty = [x["name"] for x in logs if len(x["dataframe"]) == 0]
    if empty:
        print(f"skipping {len(empty)} empty translucent log(s): {', '.join(empty)}")
    logs = [x for x in logs if len(x["dataframe"]) > 0]

    activity_number = get_activity_number(logs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    activity_number.to_csv(args.output, index=False)
    print(f"saved activity numbers of {len(activity_number)} log(s) to {args.output}")
