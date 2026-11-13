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
