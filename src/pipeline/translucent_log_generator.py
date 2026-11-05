"""
Patched copy of `create_translucent_event_log` from
external/TranslucentActivityRelationships/synthetical_log_generation/log_generator.py
(upstream commit 66bd968).

Fix: upstream hardcodes the DFA start node as frozenset({frozenset({0})}).
automata-lib 8.2.0 minimizes the DFA in `DFA.from_nfa` (minify=True by default),
which can merge the start state with an equivalent state, e.g. into
frozenset({frozenset({0}), frozenset({7})}). The hardcoded key is then missing
and the lookup raises a KeyError. Here we look up the node that contains the
initial state instead. Everything else is unchanged from upstream.
"""
import copy
import sys
from pathlib import Path

# upstream code imports `utils` / `synthetical_log_generation` as top-level packages
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "TranslucentActivityRelationships"))

import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignment
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star as star
from pm4py.objects.log.obj import EventLog

from utils.DirectReachabilityGraph import DirectReachabilityGraph
from utils.ReachabilityGraph import ReachabilityGraph
from synthetical_log_generation.log_generator import get_node_from_transition


def _find_start_node(dfa):
    # node 0 of the reachability graph is the initial marking
    initial = frozenset({0})
    for node in dfa.nodes:
        if node == initial or initial in node:
            return node
    raise ValueError("initial state not found in the DFA")


def create_translucent_event_log(log, net, im, fm, enabled_activities_name="enabled_activities",
                                 max_align_time_trace=None):
    """
    max_align_time_trace: optional time limit in seconds for aligning one variant.
    A variant whose alignment hits the limit is skipped, like a non-fitting one.
    (Added on top of upstream; None keeps upstream behaviour.)
    """
    variants = pm4py.statistics.variants.log.get.get_variants(log)
    variant_log = EventLog(attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    variant_counter = {}
    counter = 0
    for variant in variants:
        variant_log.append(variants[variant][0])
        variant_counter[counter] = len(variants[variant])
        counter += 1
    parameters = {star.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    if max_align_time_trace is not None:
        parameters[star.Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = max_align_time_trace
    rg = ReachabilityGraph(net, im, fm, 1)
    drg = DirectReachabilityGraph(rg).dfa
    start_node = _find_start_node(drg)
    global_case_id_counter = 1
    filtered_log = EventLog(attributes=log.attributes, extensions=log.extensions, classifiers=log.classifiers, omni_present=log.omni_present, properties=log.properties)
    for index, trace in enumerate(variant_log):
        node = start_node
        alignment_result = alignment.apply(trace, net, im, fm, parameters=parameters)
        # alignment_result is None when the time limit was hit
        if alignment_result is not None and alignment_result["fitness"] == 1.0:
            index_in_real_trace = 0
            for aligned_event in alignment_result["alignment"]:
                if aligned_event[1][1] is not None:
                    enabled_activities = drg.nodes[node]["enabled_activities"]
                    trace[index_in_real_trace][enabled_activities_name] = ', '.join(enabled_activities)
                    node = get_node_from_transition(drg, node, aligned_event[0][1])
                    index_in_real_trace += 1
            trace_to_append = copy.deepcopy(trace)
            counter = 0
            while counter < variant_counter[index]:
                trace_to_append._set_attributes({"concept:name": str(global_case_id_counter)})
                filtered_log.append(copy.deepcopy(trace_to_append))
                counter += 1
                global_case_id_counter += 1
    return filtered_log
