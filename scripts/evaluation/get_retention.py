"""
Share of cases that survive translucent generation, one row per log, saved as a CSV:
    cases_source   traces in the original .xes log
    cases_base     cases the translucent log was made from: the test split when the log was
                   generated with a train/test split, otherwise all traces
    cases_kept     cases in the translucent log (only perfectly fitting cases are kept)
    retention      cases_kept / cases_base

cases_kept and the split settings come from the per-log .json files that
scripts/translucent_generation/run_generation.py writes next to each result; cases_source is
counted directly in the source .xes. The test split size matches src/pipeline/split_logs.py
(train = int(n * train_ratio), test = the rest).

Usage (from the repo root):
    venv\\Scripts\\python.exe scripts\\evaluation\\get_retention.py
    venv\\Scripts\\python.exe scripts\\evaluation\\get_retention.py --translucent-dir results\\translucent_synthetic --source-dir synthetic_logs --output results\\evaluation\\synthetic\\retention.csv
    venv\\Scripts\\python.exe scripts\\evaluation\\get_retention.py --translucent-dir results\\translucent_simulated --source-dir simulated_logs --output results\\evaluation\\simulated\\retention.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRACE_TAG = re.compile(rb"<trace[\s>]")


def count_traces(xes_path: Path) -> int:
    """Number of <trace> elements in a .xes file, counted without parsing the log."""
    n = 0
    with open(xes_path, "rb") as f:
        for line in f:
            n += len(TRACE_TAG.findall(line))
    return n


def get_retention(translucent_dir: Path, source_dir: Path) -> pd.DataFrame:
    rows, missing = [], []
    for meta_path in sorted(translucent_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        name, settings = meta["name"], meta["settings"]
        source = source_dir / f"{name}.xes"
        if not source.exists():
            missing.append(name)
            continue
        cases_source = count_traces(source)
        if settings["split"]:
            cases_base = cases_source - int(cases_source * settings["train_ratio"])
        else:
            cases_base = cases_source
        rows.append({
            "name": name,
            "cases_source": cases_source,
            "cases_base": cases_base,
            "cases_kept": meta["cases"],
            "retention": meta["cases"] / cases_base if cases_base else float("nan"),
        })
    if missing:
        print(f"no source .xes in {source_dir} for {len(missing)} log(s): {', '.join(missing)}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--translucent-dir", type=Path, default=ROOT / "results" / "translucent_real_life",
                   help="folder with the per-log .json files of the translucent generation")
    p.add_argument("--source-dir", type=Path, default=ROOT / "real_life_logs",
                   help="folder with the original .xes logs")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation" / "real_life" / "retention.csv")
    args = p.parse_args()

    retention = get_retention(args.translucent_dir, args.source_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    retention.to_csv(args.output, index=False)
    print(f"saved retention of {len(retention)} log(s) to {args.output}")
