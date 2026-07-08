import pandas as pd


def measure_diversity_complexity(
    dataframe: pd.DataFrame, 
    activity_key: str, 
    enabled_activities_key: str
) -> float:
    columns = dataframe[[activity_key, enabled_activities_key]]
    counts = columns.groupby(activity_key)[enabled_activities_key].nunique()
    return counts.mean()