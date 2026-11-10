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


def compute_model_quality(models: list[Model], methods: Sequence[str] = ("token_replay",)) -> pd.DataFrame:
    """Quality of each discovered model, one row per log.

    methods selects what is computed:
      - "token_replay": precision_<v>, fitness_<v> and their F-score f_score_<v> (token-based replay)
      - "translucent_precision": translucent_precision_<v> (alignment-based, much slower)
    The F-score always uses the token-based replay values only."""
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown method(s) {sorted(unknown)}; choose from {list(METHODS)}")
    if not methods:
        raise ValueError("methods is empty")

    variants = ['IMts', 'IMto', 'IMtf', 'IMfts', 'IMfto', 'IMftf']
    rows = []

    for model in tqdm(models):
        dataframe = model['dataframe']
        row = {'name': model['name']}
        for variant in variants:
            model_var = model[variant]
            net, i_m, f_m = model_var['net'], model_var['i_m'], model_var['f_m']
            v = variant.lower()

            if "token_replay" in methods:
                precision = pm4py.precision_token_based_replay(dataframe, net, i_m, f_m)
                row[f'precision_{v}'] = precision

            if "translucent_precision" in methods:
                row[f'translucent_precision_{v}'] = _translucent_precision(dataframe, net, i_m, f_m, model['name'], variant)

            if "token_replay" in methods:
                fitness = pm4py.fitness_token_based_replay(dataframe, net, i_m, f_m)['log_fitness']
                f_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0
                row[f'fitness_{v}'] = fitness
                row[f'f_score_{v}'] = f_score
        rows.append(row)

    return pd.DataFrame(rows)
