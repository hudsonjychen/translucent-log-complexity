"""
Usage:
    python install_external_requirements.py            # install into the current interpreter
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external"
CONSTRAINTS = ROOT / "requirements_local.txt"

# submodule -> packages its used modules import (pins only for packages not in requirements_local.txt)
EXTERNAL_REQUIREMENTS = {
    # Complexity.py
    "process-complexity": [
        "BitVector==3.5.0",
        "lempel_ziv_complexity==0.2.2",
        "python-dateutil==2.9.0.post0",
        "matplotlib",
        "pandas",
        "pm4py",
    ],
    # translucent_discovery, translucent_precision, utils, synthetical_log_generation
    "TranslucentActivityRelationships": [
        "automata-lib==8.2.0",
        "networkx==2.8.5",
        "matplotlib",
        "numpy",
        "pandas",
        "pm4py",
    ],
    # distances.activity_distances (co-occurrence, context frequency, pmi)
    "semantic-aware-process-mining-distances": [
        "numpy",
        "scipy",
    ],
}


def run(cmd, dry_run):
    print(">", " ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def ensure_submodules(dry_run):
    missing = [name for name in EXTERNAL_REQUIREMENTS
               if not (EXTERNAL / name).is_dir() or not any((EXTERNAL / name).iterdir())]
    if missing:
        print(f"submodule(s) not checked out: {', '.join(missing)}")
        run(["git", "-C", str(ROOT), "submodule", "update", "--init", "--recursive"], dry_run)


def collect_packages():
    packages = {}
    for reqs in EXTERNAL_REQUIREMENTS.values():
        for req in reqs:
            name = req.split("==")[0].lower().replace("_", "-")
            # keep a pinned spec over an unpinned one
            if name not in packages or "==" in req:
                packages[name] = req
    return sorted(packages.values(), key=str.lower)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--allow-global", action="store_true",
                   help="allow installing outside a virtual environment")
    args = p.parse_args()

    in_venv = sys.prefix != sys.base_prefix
    print(f"python: {sys.executable}")
    if not in_venv and not args.allow_global and not args.dry_run:
        sys.exit("not running inside a virtual environment; activate the project venv first "
                 "or pass --allow-global to install into this interpreter anyway")

    ensure_submodules(args.dry_run)

    cmd = [sys.executable, "-m", "pip", "install", *collect_packages()]
    if CONSTRAINTS.exists():
        cmd += ["-c", str(CONSTRAINTS)]
    run(cmd, args.dry_run)
    if not args.dry_run:
        print("external requirements installed")


if __name__ == "__main__":
    main()
