from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_PATH = ROOT / "external"
sys.path.insert(0, str(EXTERNAL_PATH))

from tqdm.auto import tqdm
import pandas as pd
from .translucent_log_generator import create_translucent_event_log
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
from typing import TypedDict
from .split_logs import split_logs

class Log(TypedDict):
    name: str
    dataframe: pd.DataFrame

def generate_translucent_log(
    path_str: str,
    split: bool = True,
    noise_threshold: float = 0.4,
    train_ratio: float = 0.5,
    seed: int | None = 42,
    max_align_time_trace: float | None = None,
) -> Log:
    """Generate the translucent log for a single .xes file."""
    if split:
        # discover the model on the train part, make the test part translucent
        discovery_log, target_log = split_logs(path_str, train_ratio=train_ratio, seed=seed)
    else:
        discovery_log = pm4py.read_xes(path_str)
        discovery_log = log_converter.apply(discovery_log, variant=log_converter.Variants.TO_EVENT_LOG)
        target_log = discovery_log

    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(discovery_log, noise_threshold=noise_threshold)
    translucent_log = create_translucent_event_log(
        target_log, net, initial_marking, final_marking, max_align_time_trace=max_align_time_trace
    )

    return {
        "name": Path(path_str).stem,
        "dataframe": pm4py.convert_to_dataframe(translucent_log),
    }

def generate_translucent_logs(
    paths: list[str],
    split: bool = True,
    noise_threshold: float = 0.4,
    train_ratio: float = 0.5,
    seed: int | None = 42,
    max_align_time_trace: float | None = None,
) -> list[Log]:
    if not isinstance(paths, list):
        raise ValueError('the input should a list of str')

    logs = list()
    for path_str in tqdm(paths, desc="Log Generation"):
        if not isinstance(path_str, str) or not path_str.endswith('.xes'):
            continue
        logs.append(generate_translucent_log(
            path_str, split=split, noise_threshold=noise_threshold,
            train_ratio=train_ratio, seed=seed, max_align_time_trace=max_align_time_trace,
        ))

    return logs
