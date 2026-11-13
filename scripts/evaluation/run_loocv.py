from typing import Literal

import numpy as np
import pandas as pd
from scripts.evaluation.run_regression import run_ols


def loocv_metrics(df, predictors, target, controls: list[str] | None = None):
    model, sub = run_ols(df, predictors, target, controls)
    loo_resid = model.get_influence().resid_press
    y = sub[target].to_numpy()

    press = float((loo_resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "target": target,
        "controls": ", ".join(controls or []),
        "n": len(y),
        "R2_in": model.rsquared,
        "adjR2_in": model.rsquared_adj,
        "Q2": 1 - press / sst,
        "RMSE": np.sqrt(press / len(y)),
    }


def loocv_summary(df, targets, predictors, out_dir, type: Literal["translucent", "baseline"]="translucent",
                  controls: list[str] | None = None):
    summary = pd.DataFrame([loocv_metrics(df, predictors, t, controls) for t in targets]).set_index("target")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{type}_controlled" if controls else type
    summary.to_csv(out_dir / f"loocv_summary_{suffix}.csv")
    print(summary.round(3))
