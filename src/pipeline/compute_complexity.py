from ..complexity_measurement.measure_complexity import measure_complexity
import pandas as pd
from .generate_translucent_logs import Log
from sklearn.preprocessing import StandardScaler

def compute_complexity(logs: list[Log]) -> pd.DataFrame:
    names = list()
    complexity_avgs_raw = list()
    complexity_avgs_norm = list()
    complexity_divs = list()
    complexity_disps = list()

    for log in logs:
        name = log["name"]
        dataframe = log["dataframe"]

        names.append(name)
        complexity_avgs_raw.append(measure_complexity(dataframe, "average", avg_type="raw"))
        complexity_avgs_norm.append(measure_complexity(dataframe, "average", avg_type="norm"))
        complexity_divs.append(measure_complexity(dataframe, "diversity"))
        complexity_disps.append(measure_complexity(dataframe, "dispersion"))
    
    result_df = pd.DataFrame({
        "name": names, 
        "complexity_avg_raw": complexity_avgs_raw, 
        "complexity_avg_norm": complexity_avgs_norm, 
        "complexity_div": complexity_divs,
        "complexity_disp": complexity_disps,
    })

    std_cols = [
        "complexity_avg_raw_std", 
        "complexity_avg_norm_std", 
        "complexity_div_std", 
        "complexity_disp_std",
    ]
    
    raw_cols = [
        "complexity_avg_raw", 
        "complexity_avg_norm", 
        "complexity_div", 
        "complexity_disp", 
    ]

    scaler = StandardScaler()

    result_df[std_cols] = scaler.fit_transform(result_df[raw_cols])

    return result_df

        