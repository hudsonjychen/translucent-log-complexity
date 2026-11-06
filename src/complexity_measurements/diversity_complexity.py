import pandas as pd

SEP = ","


def measure_diversity_complexity(
    dataframe: pd.DataFrame,
    activity_key: str,
    enabled_activities_key: str
) -> float:
    # compare enabled activities as sets: the same set can be written in a different order
    enabled_sets = dataframe[enabled_activities_key].apply(
        lambda s: frozenset(a.strip() for a in str(s).split(SEP) if a.strip()) if pd.notna(s) else frozenset()
    )
    counts = enabled_sets.groupby(dataframe[activity_key]).nunique()
    return counts.mean()