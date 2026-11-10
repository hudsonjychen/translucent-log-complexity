import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.compute_model_quality import compute_model_quality
from src.pipeline.discover_petri_nets import discover_petri_nets


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "results" / "translucent_real_life" / "translucent_real_life_logs.pkl")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation" / "real_life" / "quality.csv")
    p.add_argument("--methods", nargs="+", choices=["token_replay", "translucent_precision"], default=["token_replay"],
                   help="quality measures to compute (default: token_replay)")
    p.add_argument("--timeout-hours", type=float, default=None,
                   help="time limit for translucent precision per model (one log and one IM variant); "
                        "a model over the limit gets NaN and is listed in translucent_precision_timed_out "
                        "(default: no limit)")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        logs = pickle.load(f)

    # a translucent log is empty when no test case fitted the model; there is nothing to discover from
    empty = [x["name"] for x in logs if len(x["dataframe"]) == 0]
    if empty:
        print(f"skipping {len(empty)} empty translucent log(s): {', '.join(empty)}")
    logs = [x for x in logs if len(x["dataframe"]) > 0]

    models = discover_petri_nets(logs)
    quality = compute_model_quality(models, methods=args.methods, timeout_hours=args.timeout_hours)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.output, index=False)