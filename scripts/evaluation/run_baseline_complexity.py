import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.compute_baseline_complexity import compute_baseline_complexity


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "results" / "translucent_real_life" / "translucent_real_life_logs.pkl")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation" / "real_life" / "baseline_complexity.csv")
    p.add_argument("--exp-k", type=float, default=1.0, help="coefficient for sequence entropy with exponential forgetting")
    p.add_argument("--no-zscore", action="store_true", help="do not add <metric>_z columns")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        logs = pickle.load(f)

    # a translucent log is empty when no test case fitted the model; there is nothing to measure
    empty = [x["name"] for x in logs if len(x["dataframe"]) == 0]
    if empty:
        print(f"skipping {len(empty)} empty translucent log(s): {', '.join(empty)}")
    logs = [x for x in logs if len(x["dataframe"]) > 0]

    baseline_complexity = compute_baseline_complexity(logs, exp_k=args.exp_k, zscore=not args.no_zscore)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_complexity.to_csv(args.output, index=False)
    print(f"saved baseline complexity of {len(baseline_complexity)} log(s) to {args.output}")
