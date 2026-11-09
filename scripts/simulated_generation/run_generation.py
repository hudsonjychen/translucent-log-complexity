"""
Generate random process trees, simulate a log from each, and save everything to --out-dir.

For every tree it writes:
    Simulated Log <n>.xes    the simulated log (input for the generation pipeline)
    Simulated Log <n>.ptml   the generating process tree (ground truth)
    Simulated Log <n>.pnml   the Petri net the log was simulated from
and one logs_info.json with the settings and seed of every log.

The generation itself is in src/log_simulation/generate_tree_logs.py.

Usage (from the repo root):
    venv\\Scripts\\python.exe scripts\\simulated_generation\\run_generation.py --number-per-setting 3
    venv\\Scripts\\python.exe scripts\\simulated_generation\\run_generation.py --number-per-setting 5 --parallel 0 0.1 --loop 0
    venv\\Scripts\\python.exe scripts\\simulated_generation\\run_generation.py --number-per-setting 5 --noise 0.1

--noise is optional: the share of traces per log (0..1) that get one noise operation
(remove / insert / swap an event). Without it the logs are noise-free.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pm4py
from src.log_simulation.generate_tree_logs import LOOP, PARALLEL, generate_tree_logs


def save(results: list[dict], out_dir: Path, num_traces: int, noise: float | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_info = {}
    for r in results:
        name = r["name"]
        pm4py.write_xes(r["log"], str(out_dir / f"{name}.xes"))
        pm4py.write_ptml(r["tree"], str(out_dir / f"{name}.ptml"))
        pm4py.write_pnml(r["net"], r["im"], r["fm"], str(out_dir / f"{name}.pnml"))
        logs_info[name] = {k: r[k] for k in ("parallel", "loop", "index", "seed")} | {
            "num_traces_requested": num_traces,
            "noise": noise,
            "num_traces": len(r["log"]),  # can be lower: traces that never reach the final marking are dropped
        }
        print(f"{name}: parallel={r['parallel']} loop={r['loop']} -> {len(r['log'])} traces", flush=True)

    with open(out_dir / "logs_info.json", "w", encoding="utf-8") as f:
        json.dump(logs_info, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--number-per-setting", type=int, required=True)
    p.add_argument("--parallel", type=float, nargs="+", default=list(PARALLEL))
    p.add_argument("--loop", type=float, nargs="+", default=list(LOOP))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-traces", type=int, default=1000)
    p.add_argument("--max-trace-length", type=int, default=1000)
    p.add_argument("--noise", type=float, default=None,
                   help="share of traces (0..1) that get one noise operation; default: no noise")
    p.add_argument("--out-dir", type=Path, default=ROOT / "simulated_logs")
    args = p.parse_args()

    results = generate_tree_logs(args.number_per_setting, args.parallel, args.loop, args.seed,
                                 args.num_traces, args.max_trace_length, args.noise)
    save(results, args.out_dir, args.num_traces, args.noise)
