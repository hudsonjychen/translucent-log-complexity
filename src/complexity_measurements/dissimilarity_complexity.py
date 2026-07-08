from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_PATH = ROOT / "external" / "semantic-aware-process-mining-distances"
sys.path.insert(0, str(EXTERNAL_PATH))

from typing import Literal, MutableMapping, Tuple, Dict
from collections import Counter
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from distances.activity_distances.activity_activity_co_occurence.activity_activity_co_occurrence import (
    get_activity_activity_co_occurence_matrix
)
# Upstream repo names the module activity_contex_frequency (typo: "contex").
from distances.activity_distances.activity_context_frequency.activity_contex_frequency import (
    get_activity_context_frequency_matrix,
)
from distances.activity_distances.pmi.pmi import (
    get_activity_activity_frequency_matrix_pmi,
    get_activity_context_frequency_matrix_pmi,
)


SEP = ","


DistanceKey = Tuple[str, str]
Distance = float
DistanceMatrix = Dict[DistanceKey, Distance]


def _get_distance_matrix(
    dataframe: pd.DataFrame,
    case_key: str,
    activity_key: str,
    timestamp_key: str,
    method: Literal["aa", "ac"],
    ngram_size: Literal[3, 5, 9],
    bag_of_words: bool | Literal[0, 1, 2]
) -> DistanceMatrix:
    dataframe = dataframe.sort_values([case_key, timestamp_key], kind="mergesort")
    log = (
        dataframe.groupby(case_key, sort=False)[activity_key]
        .agg(lambda acts: [str(a) for a in acts])
        .tolist()
    )
    alphabet = sorted({a for trace in log for a in trace})

    if method == "aa":
        _dm, embeddings, activity_freq_dict, activity_index = (
            get_activity_activity_co_occurence_matrix(
                log,
                alphabet,
                ngram_size,
                bag_of_words,
            )
        )
        distance_matrix_ppmi, _emb_ppmi = get_activity_activity_frequency_matrix_pmi(
            embeddings, activity_freq_dict, activity_index, 1
        )
    elif method == "ac":
        _dm, embeddings, activity_freq_dict, context_freq_dict, context_index = (
            get_activity_context_frequency_matrix(
                log,
                alphabet,
                ngram_size,
                bag_of_words,
            )
        )
        distance_matrix_ppmi, _emb_ppmi = get_activity_context_frequency_matrix_pmi(
            embeddings,
            activity_freq_dict,
            context_freq_dict,
            context_index,
            1,
        )
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'aa' or 'ac'.")

    return distance_matrix_ppmi


def _as_str_pair_key(a, b) -> DistanceKey:
    return (str(a), str(b))


def _dist(
    distance_matrix: DistanceMatrix,
    a: str,
    b: str,
    *,
    dist_cache: MutableMapping[Tuple[str, str], float],
) -> Distance:
    key, key_r = (a, b), (b, a)
    if key in dist_cache:
        return dist_cache[key]
    if key_r in dist_cache:
        return dist_cache[key_r]
    if a == b:
        d = 0.0
    else:
        if key in distance_matrix:
            d = float(distance_matrix[key])
        elif key_r in distance_matrix:
            d = float(distance_matrix[key_r])
        else:
            d = 1.0
    dist_cache[key] = d
    dist_cache[key_r] = d
    return d


def measure_dissimilarity_complexity(
    dataframe: pd.DataFrame, 
    activity_key: str,
    enabled_activities_key: str,
    case_key: str,
    timestamp_key: str,
    method: Literal["aa", "ac"],
    ngram_size: Literal[3, 5, 9],
    bag_of_words: bool | Literal[0, 1, 2]
) -> float:
    if activity_key not in dataframe.columns or enabled_activities_key not in dataframe.columns:
        raise KeyError(
            f"Expected columns {activity_key!r} and {enabled_activities_key!r} in dataframe."
        )

    df = dataframe[[activity_key, enabled_activities_key]].copy()

    enabled_lists = dataframe[enabled_activities_key].apply(
        lambda s: [a.strip() for a in str(s).split(SEP) if a.strip()] if pd.notna(s) else []
    )
    executed = {a for a in dataframe[activity_key].astype(str).str.strip() if a and a != "nan"}
    enabled_union = set().union(*enabled_lists) if len(enabled_lists) else set()
    activities = executed | enabled_union

    df["enabled_activities_list"] = enabled_lists.apply(lambda ea: tuple(sorted(ea)))

    distance_matrix = _get_distance_matrix(
        dataframe, case_key, activity_key, timestamp_key, method, ngram_size, bag_of_words
    )
    
    alphabet_list = sorted(activities)
    act_to_idx = {act: i for i, act in enumerate(alphabet_list)}
    num_acts = len(alphabet_list)
    
    dist_np = np.ones((num_acts, num_acts), dtype=float)
    np.fill_diagonal(dist_np, 0.0)
    for (a, b), d in distance_matrix.items():
        if a in act_to_idx and b in act_to_idx:
            u, v = act_to_idx[a], act_to_idx[b]
            dist_np[u, v] = float(d)
            dist_np[v, u] = float(d)

    df["enabled_activities_idx"] = df["enabled_activities_list"].apply(
        lambda ea: tuple(act_to_idx[a] for a in ea)
    )

    grouped = df.groupby(activity_key)
    per_activity_means: list[float] = []

    for a, group in grouped:
        if len(group) < 2:
            continue
            
        enabled_counts = Counter(group["enabled_activities_idx"])
        unique_sets = list(enabled_counts.keys())
        num_unique = len(unique_sets)
        
        total_pairs_sum = 0.0
        total_pairs_count = 0
        
        for i in range(num_unique):
            set_i = unique_sets[i]
            count_i = enabled_counts[set_i]
            
            total_pairs_count += count_i * (count_i - 1) // 2
            
            for j in range(i + 1, num_unique):
                set_j = unique_sets[j]
                count_j = enabled_counts[set_j]
                
                n, m = len(set_i), len(set_j)
                k = max(n, m)
                
                if k == 0:
                    dissim = 0.0
                else:
                    weight = np.ones((k, k), dtype=float)
                    
                    if n > 0 and m > 0:
                        weight[:n, :m] = dist_np[np.ix_(set_i, set_j)]
                        
                    row_ind, col_ind = linear_sum_assignment(weight)
                    num = float(weight[row_ind, col_ind].sum())
                    dissim = num / k
                
                weight_factor = count_i * count_j
                total_pairs_sum += dissim * weight_factor
                total_pairs_count += weight_factor

        if total_pairs_count > 0:
            per_activity_means.append(total_pairs_sum / total_pairs_count)

    if not per_activity_means:
        return float("nan")

    return float(np.mean(per_activity_means))
