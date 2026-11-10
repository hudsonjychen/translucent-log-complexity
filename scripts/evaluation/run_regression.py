from pathlib import Path
from typing import Literal

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def design_matrix(sub: pd.DataFrame, predictors: list[str], controls: list[str] | None = None) -> pd.DataFrame:
    """Predictors plus optional control variables, without the constant.

    Numeric controls are used as they are. Non-numeric controls (e.g. a dataset label)
    are turned into 0/1 dummies, dropping the first category as the reference."""
    controls = controls or []
    X = sub[predictors].astype(float)
    for c in controls:
        if pd.api.types.is_numeric_dtype(sub[c]):
            X[c] = sub[c].astype(float)
        else:
            X = X.join(pd.get_dummies(sub[c], prefix=c, drop_first=True, dtype=float))
    return X


def control_columns(X: pd.DataFrame, predictors: list[str]) -> list[str]:
    """Columns of the design matrix that come from the controls (dummies included)."""
    return [c for c in X.columns if c not in predictors]


def vif_table(df, predictors, controls: list[str] | None = None):
    controls = controls or []
    sub = df[predictors + controls].dropna()
    X = sm.add_constant(design_matrix(sub, predictors, controls))
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return pd.DataFrame({'variable': X.columns, 'VIF': vifs}).iloc[1:]


def run_ols(df, predictors, target, controls: list[str] | None = None):
    """OLS of target on predictors (+ optional controls). Rows with a missing value in any
    of the used columns are dropped."""
    controls = controls or []
    sub = df[predictors + controls + [target]].dropna()
    X = sm.add_constant(design_matrix(sub, predictors, controls))
    y = sub[target]
    model = sm.OLS(y, X).fit()
    return model, sub


def coef_table(model, predictors: list[str]):
    ci = model.conf_int()
    table = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p": model.pvalues,
        "ci_low": ci[0],
        "ci_high": ci[1],
    }).rename_axis("variable").reset_index()
    table["role"] = ["constant" if v == "const" else "predictor" if v in predictors else "control"
                     for v in table["variable"]]
    return table


def fit_stats(model, tgt, controls: list[str] | None = None):
    return {
        "target": tgt,
        "controls": ", ".join(controls or []),
        "n": int(model.nobs),
        "R2": model.rsquared,
        "adjR2": model.rsquared_adj,
        "F": model.fvalue,
        "F_p": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic,
    }


def run_regression(df, targets, predictors, out_dir: Path, type: Literal["translucent", "baseline", "mixed"]="translucent",
                   controls: list[str] | None = None):
    """Fit one OLS per target. `controls` is an optional list of column names that are added to
    every model as control variables; their coefficients are marked role="control" in the output.
    With controls, the files get a "_controlled" suffix so they do not overwrite the plain results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{type}_controlled" if controls else type
    all_coefs, all_stats = [], []
    for tgt in targets:
        model, sub = run_ols(df, predictors, tgt, controls)

        coefs = coef_table(model, predictors)

        all_coefs.append(coefs.assign(target=tgt))
        all_stats.append(fit_stats(model, tgt, controls))

        print(f"\n--- {tgt} ---" + (f"  (controls: {', '.join(controls)})" if controls else ""))
        print(model.summary().tables[1])
        print(f"n={int(model.nobs)}  R2={model.rsquared:.3f}  "
          f"adj.R2={model.rsquared_adj:.3f}  F p={model.f_pvalue:.4f}")

    pd.concat(all_coefs, ignore_index=True).to_csv(out_dir / f"ols_coef_all_{suffix}.csv", index=False)
    pd.DataFrame(all_stats).to_csv(out_dir / f"ols_fit_stats_{suffix}.csv", index=False)
