from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_PATH = ROOT / "external"
sys.path.insert(0, str(EXTERNAL_PATH))

from .discover_petri_nets import Model
from TranslucentActivityRelationships.translucent_precision.main import translucent_precision_score
from tqdm.notebook import tqdm
import pandas as pd
import pm4py

def compute_model_quality(models: list[Model]) -> pd.DataFrame:
    variants = ['IMts', 'IMto', 'IMtf', 'IMfts', 'IMfto', 'IMftf'] 

    precision_dict = {key: [] for key in variants}
    fitness_dict = {key: [] for key in variants}
    f_score_dict = {key: [] for key in variants}

    names = list()

    for model in tqdm(models):
        dataframe = model['dataframe']
        names.append(model['name'])
        for variant in variants:
            model_var = model[variant]
            net, i_m, f_m = model_var['net'], model_var['i_m'], model_var['f_m']

            precision = pm4py.precision_token_based_replay(dataframe, net, i_m, f_m)

            fitness_output = pm4py.fitness_token_based_replay(dataframe, net, i_m, f_m)
            fitness = fitness_output['log_fitness']

            f_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0

            precision_dict[variant].append(precision)
            fitness_dict[variant].append(fitness)
            f_score_dict[variant].append(f_score)
    
    df_dict = {'name': names}
    for variant in variants:
        df_dict[f'precision_{variant.lower()}'] = precision_dict[variant]
        df_dict[f'fitness_{variant.lower()}'] = fitness_dict[variant]
        df_dict[f'f_score_{variant.lower()}'] = f_score_dict[variant]
    return pd.DataFrame(df_dict)
