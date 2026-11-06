import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.compute_complexity import compute_complexity
from src.pipeline.compute_dissimilarity_complexity import compute_dissimilarity_complexity


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "results" / "translucent_real_life" / "translucent_real_life_logs.pkl")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation" / "real_life" / "complexity.csv")
    args = p.parse_args()

    with open(args.input, "rb") as f:
        logs = pickle.load(f)
    complexity = compute_complexity(logs)
    diss_complexity = compute_dissimilarity_complexity(logs)
    combined_complexity = complexity.merge(diss_complexity, on="name", how="inner")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined_complexity.to_csv(args.output, index=False)
