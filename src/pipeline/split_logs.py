import random

import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter


def split_logs(path: str, train_ratio: float = 0.5, seed: int | None = 42) -> tuple[EventLog, EventLog]:
    """Split the .xes log at `path` by case into (train_log, test_log)."""
    log = pm4py.read_xes(path)
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)

    traces = list(log)
    random.Random(seed).shuffle(traces)
    n_train = int(len(traces) * train_ratio)

    def make_log(subset):
        return EventLog(subset, attributes=log.attributes, extensions=log.extensions,
                        omni_present=log.omni_present, classifiers=log.classifiers,
                        properties=log.properties)

    return make_log(traces[:n_train]), make_log(traces[n_train:])
