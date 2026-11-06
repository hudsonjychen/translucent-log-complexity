"""
Generate translucent logs for every .xes file in parallel, with checkpoints.

Each log runs in its own subprocess, so a crash, out-of-memory kill or timeout
only loses that one log. Each finished log is saved right away to
<out-dir>/<name>.pkl (plus <name>.json with its settings and stats), and the
run skips logs that already have a result made with the same settings, so you
can stop and restart it at any time. When everything is done, all results are
combined into one pickle (a list of Log dicts, like generate_translucent_logs).

Usage (from the repo root):
    venv\\Scripts\\python.exe scripts\\translucent_generation\\run_generation.py --timeout-hours 3
    venv\\Scripts\\python.exe scripts\\translucent_generation\\run_generation.py --workers 12 --big-workers 2
    venv\\Scripts\\python.exe scripts\\translucent_generation\\run_generation.py --only Helpdesk "BPIC 13 Closed Problems"

    For the simulated logs, give them their own output paths so they don't mix with the real-life results:
    venv\\Scripts\\python.exe scripts\\translucent_generation\\run_generation.py --logs-dir simulated_logs --out-dir results\\translucent_simulated --combined results\\translucent_simulated\\translucent_simulated_logs.pkl

Progress:   results/logs/generation.log  (one line per start / finish / failure)
Per log:    results/logs/<name>.log      (that log's own output and traceback)
"""
import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# settings that change the result; a saved result is reused only if these match
RESULT_SETTINGS = ("split", "noise_threshold", "train_ratio", "seed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs-dir", type=Path, default=ROOT / "real_life_logs")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "translucent_real_life",
                   help="one .pkl/.json per log")
    p.add_argument("--combined", type=Path, default=ROOT / "results" / "translucent_real_life" / "translucent_real_life_logs.pkl",
                   help="combined pickle written at the end")
    p.add_argument("--only", nargs="+", metavar="NAME", help="only these logs (file names without .xes)")
    p.add_argument("--force", action="store_true", help="redo logs even if a matching result exists")

    p.add_argument("--workers", type=int, default=12, help="max logs running at once")
    p.add_argument("--big-workers", type=int, default=2, help="max big logs running at once")
    p.add_argument("--big-size-mb", type=float, default=100, help="logs at least this size count as big")
    p.add_argument("--timeout-hours", type=float, default=None,
                   help="kill a log after this many hours (default: no limit)")

    p.add_argument("--no-split", dest="split", action="store_false",
                   help="discover and generate on the whole log instead of train/test")
    p.add_argument("--noise-threshold", type=float, default=0.4)
    p.add_argument("--train-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--single", type=Path, help=argparse.SUPPRESS)  # internal: run one log
    return p.parse_args()


def settings_of(args) -> dict:
    return {k: getattr(args, k) for k in RESULT_SETTINGS}


def result_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}.pkl", out_dir / f"{name}.json"


def has_matching_result(out_dir: Path, name: str, settings: dict) -> bool:
    pkl, meta = result_paths(out_dir, name)
    if not (pkl.exists() and meta.exists()):
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("settings") == settings
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- worker mode

def run_single(args) -> int:
    """Generate one log and save it. Runs inside its own subprocess."""
    from src.pipeline.generate_translucent_logs import generate_translucent_log

    path = args.single
    name = path.stem
    pkl, meta = result_paths(args.out_dir, name)
    start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] start {name}", flush=True)
    try:
        log = generate_translucent_log(
            str(path), split=args.split, noise_threshold=args.noise_threshold,
            train_ratio=args.train_ratio, seed=args.seed,
        )
    except Exception:
        traceback.print_exc()
        return 1

    df = log["dataframe"]
    # write to a temp file first so a kill mid-write never leaves a broken .pkl
    tmp = pkl.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(log, f)
    os.replace(tmp, pkl)
    meta.write_text(json.dumps({
        "name": name,
        "settings": settings_of(args),
        "seconds": round(time.time() - start, 1),
        "cases": int(df["case:concept:name"].nunique()) if len(df) else 0,
        "events": len(df),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] done {name}: {len(df)} events", flush=True)
    return 0


# ------------------------------------------------------------- scheduler mode

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generation")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(log_dir / "generation.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def child_command(args, path: Path) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--single", str(path),
           "--out-dir", str(args.out_dir),
           "--noise-threshold", str(args.noise_threshold),
           "--train-ratio", str(args.train_ratio),
           "--seed", str(args.seed)]
    if not args.split:
        cmd.append("--no-split")
    return cmd


def child_env() -> dict:
    env = os.environ.copy()
    # one thread per process for numpy/MKL, otherwise 12 processes fight over cores
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def schedule(args) -> int:
    log_dir = args.combined.parent / "logs"
    logger = setup_logging(log_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = settings_of(args)

    paths = sorted(args.logs_dir.glob("*.xes"), key=lambda p: p.stat().st_size, reverse=True)  # biggest first
    all_paths = list(paths)  # the combined file covers every log, even when --only limits what runs
    if args.only:
        wanted = set(args.only)
        missing = wanted - {p.stem for p in paths}
        if missing:
            logger.error(f"not found in {args.logs_dir}: {sorted(missing)}")
            return 2
        paths = [p for p in paths if p.stem in wanted]

    todo = [p for p in paths if args.force or not has_matching_result(args.out_dir, p.stem, settings)]
    logger.info(f"=== run started: {len(paths)} logs, {len(paths) - len(todo)} already done, "
                f"{len(todo)} to do | workers={args.workers} big_workers={args.big_workers} | {settings}")

    big_bytes = args.big_size_mb * 1024 * 1024
    creationflags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0  # keep the PC usable
    running: dict[Path, tuple[subprocess.Popen, float, object]] = {}
    failed: list[str] = []
    pending = list(todo)

    try:
        while pending or running:
            # start new logs while there is room
            n_big = sum(p.stat().st_size >= big_bytes for p in running)
            for path in list(pending):
                if len(running) >= args.workers:
                    break
                is_big = path.stat().st_size >= big_bytes
                if is_big and n_big >= args.big_workers:
                    continue
                out = open(log_dir / f"{path.stem}.log", "w", encoding="utf-8")
                proc = subprocess.Popen(child_command(args, path), cwd=ROOT, env=child_env(),
                                        stdout=out, stderr=subprocess.STDOUT, creationflags=creationflags)
                running[path] = (proc, time.time(), out)
                pending.remove(path)
                n_big += is_big
                logger.info(f"start  {path.stem} ({path.stat().st_size / 2**20:.0f} MB)  "
                            f"[running {len(running)}, waiting {len(pending)}]")

            time.sleep(5)

            # collect finished / timed-out logs
            for path, (proc, started, out) in list(running.items()):
                elapsed = time.time() - started
                code = proc.poll()
                if code is None and args.timeout_hours and elapsed > args.timeout_hours * 3600:
                    proc.kill()
                    proc.wait()
                    code = "timeout"
                if code is None:
                    continue
                out.close()
                del running[path]
                if code == 0:
                    logger.info(f"done   {path.stem} in {fmt_duration(elapsed)}")
                else:
                    failed.append(path.stem)
                    reason = "timed out" if code == "timeout" else f"exit code {code}"
                    logger.error(f"FAILED {path.stem} after {fmt_duration(elapsed)} ({reason}); "
                                 f"see {log_dir / (path.stem + '.log')}")
    except KeyboardInterrupt:
        logger.warning("interrupted, stopping running logs (finished results are kept)")
        for proc, _, out in running.values():
            proc.kill()
            out.close()
        return 130

    # combine every log in --logs-dir that has a matching result (not only the ones run now)
    done = [p for p in sorted(all_paths, key=lambda p: p.stem) if has_matching_result(args.out_dir, p.stem, settings)]
    combined = []
    for p in done:
        with open(result_paths(args.out_dir, p.stem)[0], "rb") as f:
            log = pickle.load(f)
        log["name"] = p.stem  # the file name wins, so renamed logs keep their new name
        combined.append(log)
    args.combined.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.combined.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(combined, f)
    os.replace(tmp, args.combined)

    n_done_now = sum(has_matching_result(args.out_dir, p.stem, settings) for p in paths)
    logger.info(f"=== run finished: {n_done_now}/{len(paths)} logs done, {len(failed)} failed"
                f"{' (' + ', '.join(failed) + ')' if failed else ''}; "
                f"combined {len(done)}/{len(all_paths)} logs -> {args.combined}")
    return 1 if failed else 0


if __name__ == "__main__":
    arguments = parse_args()
    sys.exit(run_single(arguments) if arguments.single else schedule(arguments))
