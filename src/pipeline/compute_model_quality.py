from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_PATH = ROOT / "external"
sys.path.insert(0, str(EXTERNAL_PATH))

from .discover_petri_nets import Model
from TranslucentActivityRelationships.translucent_precision.main import get_alignments, compute_translucent_precision
from TranslucentActivityRelationships.translucent_precision.AlignmentEnabledPrefixAutomaton import AlignmentEnabledPrefixAutomaton
from TranslucentActivityRelationships.utils.ReachabilityGraph import ReachabilityGraph
from TranslucentActivityRelationships.utils.DirectReachabilityGraph import DirectReachabilityGraph
from pm4py.objects.conversion.log import converter as log_converter
from tqdm.auto import tqdm
import math
import multiprocessing as mp
import queue
import time
from typing import Sequence
import networkx as nx
import pandas as pd
import pm4py

METHODS = ("token_replay", "translucent_precision")

# start node of the model DFA as compute_translucent_precision expects it
DFA_START = frozenset({frozenset({0})})


def _translucent_precision(dataframe: pd.DataFrame, net, i_m, f_m, name: str, variant: str) -> float:
    """Translucent precision, as translucent_precision_score in TranslucentActivityRelationships.

    Same steps as upstream, with one fix: upstream looks up the start of the model DFA as
    DFA_START, but automata-lib can merge the start state with equivalent states (e.g. when the
    model can return to its initial state), and the start node is then named
    frozenset({frozenset({0}), ...}). Equivalent states have the same enabled activities, so the
    merged node is renamed to DFA_START. Only perfectly fitting traces are used; with none the
    score is undefined (NaN)."""
    log = log_converter.apply(dataframe, variant=log_converter.Variants.TO_EVENT_LOG)
    alignments = get_alignments(log, net, i_m, f_m)
    automaton = AlignmentEnabledPrefixAutomaton(log, net, i_m, f_m).alignment_enabled_prefix_automaton
    r_g = ReachabilityGraph(net, i_m, f_m, 1)
    d_r_g = DirectReachabilityGraph(r_g)
    dfa = d_r_g.dfa
    if DFA_START not in dfa:
        (start,) = [node for node in dfa if frozenset({0}) in node]
        nx.relabel_nodes(dfa, {start: DFA_START}, copy=False)
    try:
        return compute_translucent_precision(alignments, automaton, r_g, d_r_g.direct_reachability_graph, dfa)
    except ZeroDivisionError:
        tqdm.write(f"{name} / {variant}: no perfectly fitting trace, translucent precision is NaN")
        return math.nan


def _translucent_precision_worker(jobs: list, results) -> None:
    """Child process: compute translucent precision for each job in order and report each result."""
    for variant, dataframe, net, i_m, f_m, name in jobs:
        results.put((variant, _translucent_precision(dataframe, net, i_m, f_m, name, variant)))


def _translucent_precisions_with_timeout(model: Model, variants: Sequence[str], timeout_s: float) -> tuple[dict, list]:
    """Translucent precision of each variant of one log, each limited to timeout_s seconds.

    The variants run one after another in a child process. When one runs over the limit (or the
    child crashes), the child is killed, that variant gets NaN, and a new child continues with the
    remaining variants. Returns ({variant: value}, [variants that timed out or failed])."""
    ctx = mp.get_context("spawn")
    values, failed = {}, []
    pending = list(variants)
    while pending:
        jobs = [(v, model['dataframe'], model[v]['net'], model[v]['i_m'], model[v]['f_m'], model['name'])
                for v in pending]
        results = ctx.Queue()
        worker = ctx.Process(target=_translucent_precision_worker, args=(jobs, results), daemon=True)
        worker.start()
        deadline = time.monotonic() + timeout_s
        stuck = None
        while pending:
            try:
                variant, value = results.get(timeout=max(0.0, min(5.0, deadline - time.monotonic())))
            except queue.Empty:
                if not worker.is_alive():
                    stuck = "crashed"
                    break
                if time.monotonic() >= deadline:
                    stuck = "timed out"
                    break
                continue
            values[variant] = value
            pending.remove(variant)
            deadline = time.monotonic() + timeout_s  # the next variant gets its own full limit
        if stuck:
            variant = pending.pop(0)  # the child works through the jobs in order
            values[variant] = math.nan
            failed.append(variant)
            tqdm.write(f"{model['name']} / {variant}: translucent precision {stuck} "
                       f"(limit {timeout_s / 3600:g} h), set to NaN")
        if worker.is_alive():
            worker.terminate()
        worker.join()
    return values, failed


def compute_model_quality(models: list[Model], methods: Sequence[str] = ("token_replay",),
                          timeout_hours: float | None = None) -> pd.DataFrame:
    """Quality of each discovered model, one row per log.

    methods selects what is computed:
      - "token_replay": precision_<v>, fitness_<v> and their F-score f_score_<v> (token-based replay)
      - "translucent_precision": translucent_precision_<v> (alignment-based, much slower)
    The F-score always uses the token-based replay values only.

    timeout_hours limits translucent precision per model (one log and one variant). A model that runs
    over the limit gets NaN and is listed in the translucent_precision_timed_out column. Without a
    limit everything runs in this process, as before."""
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown method(s) {sorted(unknown)}; choose from {list(METHODS)}")
    if not methods:
        raise ValueError("methods is empty")
    if timeout_hours is not None and timeout_hours <= 0:
        raise ValueError("timeout_hours must be positive")

    variants = ['IMts', 'IMto', 'IMtf', 'IMfts', 'IMfto', 'IMftf']
    rows = []

    for model in tqdm(models):
        dataframe = model['dataframe']
        row = {'name': model['name']}

        translucent = {}
        if "translucent_precision" in methods:
            if timeout_hours is None:
                translucent = {variant: _translucent_precision(dataframe, model[variant]['net'], model[variant]['i_m'],
                                                               model[variant]['f_m'], model['name'], variant)
                               for variant in variants}
            else:
                translucent, failed = _translucent_precisions_with_timeout(model, variants, timeout_hours * 3600)
                row['translucent_precision_timed_out'] = ", ".join(failed)

        for variant in variants:
            model_var = model[variant]
            net, i_m, f_m = model_var['net'], model_var['i_m'], model_var['f_m']
            v = variant.lower()

            if "token_replay" in methods:
                precision = pm4py.precision_token_based_replay(dataframe, net, i_m, f_m)
                row[f'precision_{v}'] = precision

            if "translucent_precision" in methods:
                row[f'translucent_precision_{v}'] = translucent[variant]

            if "token_replay" in methods:
                fitness = pm4py.fitness_token_based_replay(dataframe, net, i_m, f_m)['log_fitness']
                f_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0
                row[f'fitness_{v}'] = fitness
                row[f'f_score_{v}'] = f_score
        rows.append(row)

    return pd.DataFrame(rows)
