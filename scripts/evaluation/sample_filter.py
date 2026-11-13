from typing import Callable, Sequence

import pandas as pd


def apply_two_part_rule(df: pd.DataFrame, min_cases: int = 30, min_retention: float = 0.10,
                        cases_col: str = "support", retention_col: str = "retention"
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    too_small = df[cases_col] < min_cases
    unknown = df[retention_col].isna()
    low_retention = df[retention_col] < min_retention

    reasons = pd.Series("", index=df.index)
    reasons[too_small] += f"< {min_cases} cases; "
    reasons[low_retention] += f"< {min_retention:.0%} kept; "
    reasons[unknown] += "retention unknown; "
    reasons = reasons.str.rstrip("; ")

    drop = too_small | low_retention | unknown
    kept = df[~drop].reset_index(drop=True)
    excluded = df[drop].assign(reason=reasons[drop]).reset_index(drop=True)
    return kept, excluded


def sensitivity_grid(df: pd.DataFrame, case_thresholds: Sequence[int], retention_thresholds: Sequence[float],
                     analysis: Callable[[pd.DataFrame], dict], **rule_kwargs) -> pd.DataFrame:
    rows = []
    for min_cases in case_thresholds:
        for min_retention in retention_thresholds:
            kept, _ = apply_two_part_rule(df, min_cases, min_retention, **rule_kwargs)
            rows.append({"min_cases": min_cases, "min_retention": min_retention, "n": len(kept), **analysis(kept)})
    return pd.DataFrame(rows)
