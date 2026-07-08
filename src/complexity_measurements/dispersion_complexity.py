import numpy as np
import pandas as pd

SEP = ","

def measure_dispersion_complexity(
    dataframe: pd.DataFrame, 
    activity_key: str, 
    enabled_activities_key: str
) -> float:
    if activity_key not in dataframe.columns or enabled_activities_key not in dataframe.columns:
        raise KeyError(f"Expected columns {activity_key!r} and {enabled_activities_key!r} in dataframe.")

    enabled_lists = dataframe[enabled_activities_key].apply(
        lambda s: [a.strip() for a in str(s).split(SEP) if a.strip()] if pd.notna(s) else []
    )
    
    executed = {a for a in dataframe[activity_key].astype(str).str.strip() if a and a != "nan"}
    
    enabled_union = set().union(*enabled_lists) if len(enabled_lists) else set()
    activities = executed | enabled_union

    if not activities:
        return float("nan")
        
    alphabet_list = sorted(activities)
    num_activities = len(alphabet_list)
    act_to_idx = {act: i for i, act in enumerate(alphabet_list)}

    df_clean = pd.DataFrame({
        "activity": dataframe[activity_key],
        "enabled_list": enabled_lists
    }).reset_index(drop=True)

    exploded = df_clean["enabled_list"].explode().dropna()
    
    binary_matrix = np.zeros((len(df_clean), num_activities), dtype=np.int8)
    if not exploded.empty:
        row_indices = exploded.index.values
        col_indices = exploded.map(act_to_idx).values.astype(int)
        binary_matrix[row_indices, col_indices] = 1

    grouped_indices = df_clean.groupby("activity").indices
    per_activity_disp = []
    sqrt_num_acts = np.sqrt(num_activities)

    for activity, indices in grouped_indices.items():
        if len(indices) == 0:
            continue
            
        vector = binary_matrix[indices]
        mean_vec = vector.mean(axis=0)
        
        disp = np.mean(np.sqrt(np.sum((vector - mean_vec) ** 2, axis=1))) / sqrt_num_acts
        per_activity_disp.append(disp)

    if not per_activity_disp:
        return float("nan")

    return float(np.mean(per_activity_disp))