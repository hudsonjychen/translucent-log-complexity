"""
Split BPIC 19 into sub-logs by item category (case:Item Category).

The BPIC 2019 process consists of four flows, one per item category:
    2-way match, 3-way match invoice after GR, 3-way match invoice before GR, Consignment

Each becomes its own .xes file named "BPIC 19 <category>.xes".
"""
import argparse
from pathlib import Path

import pm4py

ROOT = Path(__file__).resolve().parent.parent.parent
ATTRIBUTE = "case:Item Category"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "real_life_logs" / "BPIC 19.xes")
    p.add_argument("--out-dir", type=Path, default=ROOT / "real_life_logs")
    args = p.parse_args()

    df = pm4py.read_xes(str(args.input))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for category, sub in df.groupby(ATTRIBUTE):
        name = "BPIC 19 " + category.replace(",", "")
        out = args.out_dir / f"{name}.xes"
        pm4py.write_xes(sub, str(out))
        print(f"{name}: {sub['case:concept:name'].nunique()} cases, {len(sub)} events -> {out}")


if __name__ == "__main__":
    main()
