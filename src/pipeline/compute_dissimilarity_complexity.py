from typing import Dict

from ..complexity_measurement.measure_complexity import measure_complexity
from tqdm.notebook import tqdm
import pandas as pd
from .generate_translucent_logs import Log
from sklearn.preprocessing import StandardScaler

def compute_dissimilarity_complexity(logs: list[Log]) -> pd.DataFrame:
    names = list()
    aa_seq, aa_mset, ac_seq, ac_mset = list(), list(), list(), list()
    params: Dict[tuple, list] = {
        ("aa", False): aa_seq, 
        ("aa", True): aa_mset, 
        ("ac", 0): ac_seq, 
        ("ac", 2): ac_mset
    }

    for log in tqdm(logs):
        name = log["name"]
        dataframe = log["dataframe"]

        names.append(name)
        for param, l in params.items():
            method, bag_of_words = param
            l.append(measure_complexity(
                dataframe,
                "dissimilarity",
                method=method,
                bag_of_words=bag_of_words
            ))
    
    result_df = pd.DataFrame({
        "name": names, 
        "complexity_diss_aa_seq": aa_seq,
        "complexity_diss_aa_mset": aa_mset,
        "complexity_diss_ac_seq": ac_seq,
        "complexity_diss_ac_mset": ac_mset
    })

    std_cols = [
        "complexity_diss_aa_seq_std", 
        "complexity_diss_aa_mset_std", 
        "complexity_diss_ac_seq_std", 
        "complexity_diss_ac_mset_std"
    ]
    
    raw_cols = [
        "complexity_diss_aa_seq", 
        "complexity_diss_aa_mset", 
        "complexity_diss_ac_seq", 
        "complexity_diss_ac_mset"
    ]

    scaler = StandardScaler()

    result_df[std_cols] = scaler.fit_transform(result_df[raw_cols])

    return result_df 