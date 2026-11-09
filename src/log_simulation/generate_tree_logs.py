"""
Random process trees -> Petri nets -> simulated event logs.

Only generation lives here; saving the results is up to the caller
(see scripts/simulated_logs/run_generation.py).
"""
import copy
import random

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import ProcessTree

PARALLEL = (0.05, 0.1, 0.25)
LOOP = (0.05, 0.1, 0.25)


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


NOISE_OPERATIONS = ("remove", "insert", "swap")


def add_noise(log: EventLog, noise: float) -> EventLog:
    """
    Make a share `noise` of the traces noisy, in place. Each chosen trace gets one random operation:
        remove  delete a random event (only if the trace has more than one event)
        insert  insert a random activity of the log at a random position
        swap    swap the activities of two neighbouring events
    Only activity names move, so timestamps stay in order.
    Empty traces (a tree that can skip everything) are left alone; the share is taken over the non-empty ones.
    """
    activities = sorted({e["concept:name"] for t in log for e in t})
    traces = [t for t in log if len(t) > 0]
    for trace in random.sample(traces, round(noise * len(traces))):
        op = random.choice(NOISE_OPERATIONS)
        if op == "remove" and len(trace) > 1:
            del trace._list[random.randrange(len(trace))]
        elif op == "swap" and len(trace) > 1:
            i = random.randrange(len(trace) - 1)
            trace[i]["concept:name"], trace[i + 1]["concept:name"] = trace[i + 1]["concept:name"], trace[i]["concept:name"]
        else:  # insert, also the fallback for one-event traces
            i = random.randrange(len(trace) + 1)
            event = copy.deepcopy(trace[min(i, len(trace) - 1)])
            event["concept:name"] = random.choice(activities)
            trace.insert(i, event)
    return log


def generate_tree_logs(
    number_per_setting: int,
    parallel=PARALLEL,
    loop=LOOP,
    seed: int = 42,
    num_traces: int = 1000,
    max_trace_length: int = 1000,
    noise: float | None = None,
) -> list[dict]:
    """
    Generate trees, convert each to a Petri net and simulate a log from the net.
    If `noise` (0..1) is given, that share of the traces in every log gets one noise operation (see add_noise).

    Returns one dict per tree with keys:
        name, parallel, loop, index, seed, tree, net, im, fm, log
    """
    results = list()
    for n, t in enumerate(generate_process_trees(number_per_setting, parallel, loop, seed)):
        net, im, fm = pm4py.convert_to_petri_net(t["tree"])
        log = simulate_log(net, im, fm, num_traces, max_trace_length)
        if noise:
            add_noise(log, noise)
        results.append({"name": f"Simulated Log {n+1}", **t, "net": net, "im": im, "fm": fm, "log": log})
    return results
