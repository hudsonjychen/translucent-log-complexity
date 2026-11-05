"""
Random process trees -> Petri nets -> simulated event logs.

Only generation lives here; saving the results is up to the caller
(see scripts/simulated_logs/run_generation.py).
"""
import random

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import ProcessTree

PARALLEL = (0, 0.1, 0.25)
LOOP = (0, 0.1, 0.25)


def generate_process_trees(number_per_setting: int, parallel=PARALLEL, loop=LOOP, seed: int = 42) -> list[dict]:
    """Generate `number_per_setting` trees for every (parallel, loop) combination."""
    trees = list()
    for p in parallel:
        for l in loop:
            rest = (1 - p - l) / 2
            for i in range(number_per_setting):
                tree_seed = seed + len(trees)
                random.seed(tree_seed)
                np.random.seed(tree_seed)
                tree: ProcessTree = pm4py.generate_process_tree(parameters={
                    "parallel": p, "loop": l, "sequence": rest, "choice": rest,
                })
                trees.append({"parallel": p, "loop": l, "index": i, "seed": tree_seed, "tree": tree})
    return trees


def simulate_log(net, im, fm, num_traces: int, max_trace_length: int) -> EventLog:
    """Play out the Petri net with pm4py. Case ids and timestamps are added by the playout."""
    return pm4py.play_out(net, im, fm, parameters={
        "noTraces": num_traces,
        "maxTraceLength": max_trace_length,
        # only keep traces that reach the final marking, so no cut-off traces end up in the log
        "add_only_if_fm_is_reached": True,
    })


def generate_tree_logs(
    number_per_setting: int,
    parallel=PARALLEL,
    loop=LOOP,
    seed: int = 42,
    num_traces: int = 1000,
    max_trace_length: int = 1000,
) -> list[dict]:
    """
    Generate trees, convert each to a Petri net and simulate a log from the net.

    Returns one dict per tree with keys:
        name, parallel, loop, index, seed, tree, net, im, fm, log
    """
    results = list()
    for n, t in enumerate(generate_process_trees(number_per_setting, parallel, loop, seed)):
        net, im, fm = pm4py.convert_to_petri_net(t["tree"])
        log = simulate_log(net, im, fm, num_traces, max_trace_length)
        results.append({"name": f"Simulated Log {n}", **t, "net": net, "im": im, "fm": fm, "log": log})
    return results
