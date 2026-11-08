"""
Create a new combined pickle without some of its logs.

Reads a combined pickle (a list of {"name", "dataframe"} dicts, as written by run_generation.py),
drops logs by the filters below, and writes the rest to a new file. The input file is never changed.

Filters (use one or both; a log is removed if any filter matches):
    --remove NAME [NAME ...]        remove these logs by name
    --case-number-threshold N       remove logs with fewer than N cases (empty logs count as 0)

Usage (from the repo root):
    venv\\Scripts\\python.exe scripts\\remove_logs.py --remove "BPIC 13 Incidents" Sepsis
    venv\\Scripts\\python.exe scripts\\remove_logs.py --case-number-threshold 50
    venv\\Scripts\\python.exe scripts\\remove_logs.py ^
        --input results\\translucent_simulated\\translucent_simulated_logs.pkl ^
        --output results\\translucent_simulated\\translucent_simulated_logs_min50.pkl ^
        --case-number-threshold 50 --remove "Simulated Log 3"
"""
import argparse
import difflib
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = "case:concept:name"


def n_cases(log: dict) -> int:
    df = log["dataframe"]
    return int(df[CASE].nunique()) if len(df) and CASE in df.columns else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=ROOT / "results" / "translucent_real_life" / "translucent_real_life_logs.pkl")
    p.add_argument("--output", type=Path, default=None,
                   help="default: <input name>_reduced.pkl next to the input")
    p.add_argument("--remove", nargs="+", default=[], metavar="NAME", help="log names to leave out")
    p.add_argument("--case-number-threshold", type=int, default=None, metavar="N",
                   help="leave out logs with fewer than N cases")
    args = p.parse_args()

    if not args.remove and args.case_number_threshold is None:
        print("error: give --remove and/or --case-number-threshold")
        return 2
    output = args.output or args.input.with_name(f"{args.input.stem}_reduced.pkl")
    if output.resolve() == args.input.resolve():
        print("error: --output must differ from --input (the input file is never overwritten)")
        return 2

    with open(args.input, "rb") as f:
        logs = pickle.load(f)
    names = [x["name"] for x in logs]

    unknown = [n for n in args.remove if n not in names]
    if unknown:
        for n in unknown:
            close = difflib.get_close_matches(n, names, n=3)
            print(f"error: no log named {n!r} in {args.input.name}"
                  + (f" (did you mean: {', '.join(close)}?)" if close else ""))
        return 2

    # why each log is removed (a log can match both filters)
    reasons: dict[str, list[str]] = {}
    for n in args.remove:
        reasons.setdefault(n, []).append("named in --remove")
    if args.case_number_threshold is not None:
        for log in logs:
            c = n_cases(log)
            if c < args.case_number_threshold:
                reasons.setdefault(log["name"], []).append(f"{c} cases < {args.case_number_threshold}")

    kept = [x for x in logs if x["name"] not in reasons]

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(kept, f)
    os.replace(tmp, output)

    print(f"removed {len(reasons)}:")
    for name in sorted(reasons):
        print(f"  {name}  ({'; '.join(reasons[name])})")
    print(f"kept {len(kept)} of {len(logs)} logs -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
