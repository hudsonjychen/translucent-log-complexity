from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def get_corr_matrix(df, method: Literal["pearson", "kendall"] = "pearson"):
    df = df.select_dtypes(include=[np.number])
    return df.corr(method=method)


def get_p_values(df, method: Literal["pearson", "kendall"] = "pearson"):
    df = df.select_dtypes(include=[np.number])
    test = {"pearson": pearsonr, "kendall": kendalltau}[method]
    cols = df.columns
    p_val_df = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            # drop missing values per pair, like df.corr does
            pair = df.iloc[:, [i, j]].dropna()
            p_val_df.iloc[i, j] = test(pair.iloc[:, 0], pair.iloc[:, 1])[1]

    return p_val_df


def show_heatmap(corr, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
    )
    
    plt.title(title)
    plt.show()


def show_scatterplot(df, x, y, title, xlabel = None, ylabel = None):
    xlabel = xlabel or x
    ylabel = ylabel or y

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        palette="deep",
        s=100,
    )
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def _with_type(path: Path, type: str) -> Path:
    """p_corr.csv -> p_corr_translucent.csv"""
    path = Path(path)
    return path.with_name(f"{path.stem}_{type}{path.suffix}")


def run_correlations(df, outputs: list[Path], type: Literal["translucent", "baseline"] = "translucent"):
    """outputs: [pearson corr, pearson p-values, kendall corr, kendall p-values];
    each file is saved with _<type> before its extension. Saved without row labels
    (the rows are in the same order as the columns). verbose=True prints each written path."""
    p_corr = get_corr_matrix(df, method="pearson")
    p_p_values = get_p_values(df, method="pearson")
    k_corr = get_corr_matrix(df, method="kendall")
    k_p_values = get_p_values(df, method="kendall")

    for result, output in zip((p_corr, p_p_values, k_corr, k_p_values), outputs[:4]):
        output = _with_type(output, type)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)

    return p_corr, p_p_values, k_corr, k_p_values