from typing import Literal
import pandas as pd

from .average_complexity import measure_average_complexity
from .diversity_complexity import measure_diversity_complexity
from .dispersion_complexity import measure_dispersion_complexity
from .dissimilarity_complexity import measure_dissimilarity_complexity


ENABLED_ACTIVITIES = 'enabled_activities'
CASE = 'case:concept:name'
ACTIVITY = 'concept:name'
TIMESTAMP = 'time:timestamp'


def measure_complexity(
    dataframe: pd.DataFrame, 
    variant: Literal["average", "diversity", "dispersion", "dissimilarity"], 
    case_key: str = CASE,
    activity_key: str = ACTIVITY,
    enabled_activities_key: str = ENABLED_ACTIVITIES,
    timestamp_key: str = TIMESTAMP,
    *,
    avg_type: Literal["raw", "norm"] = "raw",
    method: Literal["aa", "ac"] = "aa",
    ngram_size: Literal[3, 5, 9] = 3,
    bag_of_words: bool = False
) -> float:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe must be a pandas DataFrame, got {type(dataframe).__name__}")
    if variant == "average":
        return measure_average_complexity(
            dataframe, 
            activity_key,
            enabled_activities_key,
            avg_type
        )
    elif variant == "diversity":
        return measure_diversity_complexity(
            dataframe, 
            activity_key, 
            enabled_activities_key
        )
    elif variant == "dispersion":
        return measure_dispersion_complexity(
            dataframe, 
            activity_key, 
            enabled_activities_key
        )
    elif variant == "dissimilarity":
        return measure_dissimilarity_complexity(
            dataframe, 
            activity_key, 
            enabled_activities_key, 
            case_key, 
            timestamp_key, 
            method, 
            ngram_size, 
            bag_of_words
        )
    else:
        raise ValueError(
            f"variant must be one of 'average', 'diversity', 'dispersion', "
            f"'dissimilarity', got {variant!r}"
        )