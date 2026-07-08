from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_PATH = ROOT / "external"
sys.path.insert(0, str(EXTERNAL_PATH))

from .generate_translucent_logs import Log
from TranslucentActivityRelationships.translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
import pm4py
from pm4py import PetriNet, Marking
import pandas as pd
from typing import TypedDict

class Variant(TypedDict):
    net: PetriNet
    i_m: Marking
    f_m: Marking

class Model(TypedDict):
    name: str
    dataframe: pd.DataFrame
    IMts: Variant
    IMto: Variant
    IMtf: Variant
    IMfts: Variant
    IMfto: Variant
    IMftf: Variant

def discover_petri_nets(logs: list[Log], noise_threshold: float | int = 0.2) -> list[Model]:
    variants = ['IMts', 'IMto', 'IMtf'] 
    variants_with_noise_threshold = ['IMfts', 'IMfto', 'IMftf'] 
    models = list() 
    for log in logs:
        dataframe = log['dataframe']
        formatted_dataframe = pm4py.format_dataframe(
            dataframe, 
            case_id='case:concept:name', 
            activity_key='concept:name', 
            timestamp_key='time:timestamp'
        )
        event_log = pm4py.convert_to_event_log(formatted_dataframe)

        model = {
            'name': log['name'],
            'dataframe': log['dataframe']
        }
        for variant in variants:
            net, i_m, f_m = discover_petri_net(event_log, {"translucent_variant": variant, "tDFG_fall_through": False})
            model[variant] = {'net': net, 'i_m': i_m, 'f_m': f_m}
        
        for index, variant in enumerate(variants_with_noise_threshold):
            net, i_m, f_m = discover_petri_net(event_log, {"translucent_variant": variants[index], "tDFG_fall_through": False}, noise_threshold)
            model[variant] = {'net': net, 'i_m': i_m, 'f_m': f_m}
        
        models.append(model)
    
    return models
        