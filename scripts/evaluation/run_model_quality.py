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
    args = p.parse_args()

    with open(args.input, "rb") as f:
        logs = pickle.load(f)

    models = discover_petri_nets(logs)
    quality = compute_model_quality(models)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(args.output, index=False)