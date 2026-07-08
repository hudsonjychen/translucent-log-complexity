from typing import Literal
import pandas as pd
import numpy as np

SEP = ","


def measure_average_complexity(
    dataframe: pd.DataFrame, 
    activity_key: str,
    enabled_activities_key: str,
    type: Literal["raw", "norm"] = "raw",
) -> float:
    
    enabled_lists = dataframe[enabled_activities_key].apply(
        lambda s: [a.strip() for a in str(s).split(SEP) if a.strip()] if pd.notna(s) else []
    )
    executed = {a for a in dataframe[activity_key].astype(str).str.strip() if a and a != "nan"}
    enabled_union = set().union(*enabled_lists) if len(enabled_lists) else set()
    activities = executed | enabled_union

    if not activities:
        return float("nan")

    mean_enabled = enabled_lists.apply(lambda x: len(set(x))).mean()

    if type == "raw":
        return mean_enabled
    elif type == "norm":
        if len(activities) <= 1:
            return float("nan")
        return (mean_enabled - 1) / (len(activities) - 1)
    else:
        raise ValueError(f"type must be 'raw' or 'norm', got {type!r}")
