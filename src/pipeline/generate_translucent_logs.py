from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_PATH = ROOT / "external"
sys.path.insert(0, str(EXTERNAL_PATH))

from tqdm.notebook import tqdm
import pandas as pd
from TranslucentActivityRelationships.synthetical_log_generation.log_generator import create_translucent_event_log
from pm4py.objects.conversion.log import converter as log_converter
import pm4py
from typing import TypedDict

class Log(TypedDict):
    name: str
    dataframe: pd.DataFrame

def generate_translucent_logs(paths: list[str]) -> list[Log]:
    if not isinstance(paths, list):
        raise ValueError('the input should a list of str')
    
    logs = list()
    
    for path_str in tqdm(paths, desc="Log Generation"):
        if not isinstance(path_str, str) or not path_str.endswith('.xes'):
            continue

        initial_log = pm4py.read_xes(path_str)
        initial_log = log_converter.apply(initial_log, variant=log_converter.Variants.TO_EVENT_LOG)
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(initial_log, noise_threshold=0.4)
        translucent_log = create_translucent_event_log(initial_log, net, initial_marking, final_marking)
        
        dataframe = pm4py.convert_to_dataframe(translucent_log)
        name = Path(path_str).stem

        logs.append({
            "name": name,
            "dataframe": dataframe
        })

    return logs