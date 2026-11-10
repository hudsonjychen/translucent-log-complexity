"""
Heatmaps of complexity measures vs. F-score per discovery variant, Pearson and Kendall side by side.

Reads <corr-dir>/{p,k}_corr.csv and {p,k}_p_values.csv (square matrices saved without row labels,
rows in the same order as the columns) and writes complexity_fscore_heatmap.png / .pdf next to them.

Each cell shows the coefficient, marked with * (and bold) when p < 0.05.

Usage (from the repo root):
    venv\\Scripts\\python.exe scripts\\evaluation\\plot_corr_heatmap.py
    venv\\Scripts\\python.exe scripts\\evaluation\\plot_corr_heatmap.py --corr-dir results\\evaluation\\group1_corr --n 39
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]

ROWS = {
    "complexity_avg_raw": "Average (raw)",
    "complexity_avg_norm": "Average (normalized)",
    "complexity_div": "Diversity",
    "complexity_disp": "Dispersion",
    "complexity_diss_aa_seq": "Dissimilarity (aa, seq)",
}
COLS = {
    "f_score_imtf": "IMtf",
    "f_score_imts": "IMts",
    "f_score_imto": "IMto",
    "f_score_imftf": "IMftf",
    "f_score_imfts": "IMfts",
    "f_score_imfto": "IMfto",
}

# diverging: blue (negative) <- neutral gray -> red (positive), equal steps per arm
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#104281", "#256abf", "#6da7ec", "#b7d3f6", "#f0efec", "#f6c3c0", "#ee8a87", "#e34948", "#a12c2c"],
)
INK, INK_MUTED, INK_ON_DARK = "#1f1f1e", "#6b6a66", "#ffffff"
SURFACE = "#fcfcfb"


def load(corr_dir: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = pd.read_csv(corr_dir / f"{prefix}_corr.csv")
    pval = pd.read_csv(corr_dir / f"{prefix}_p_values.csv")
    corr.index, pval.index = corr.columns, pval.columns  # saved without row labels
    return corr.loc[list(ROWS), list(COLS)], pval.loc[list(ROWS), list(COLS)]


def draw(ax, corr: pd.DataFrame, pval: pd.DataFrame, title: str, vlim: float):
    values = corr.to_numpy()
    im = ax.imshow(values, cmap=DIVERGING, vmin=-vlim, vmax=vlim, aspect="auto")

    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v, p = values[i, j], pval.iat[i, j]
            mark = "*" if p < 0.05 else ""
            color = INK_ON_DARK if abs(v) > 0.55 * vlim else INK
            ax.text(j, i, f"{v:+.2f}{mark}".replace("-", "−"), ha="center", va="center", fontsize=10, color=color,
                    fontweight="bold" if mark else "normal")

    ax.set_xticks(range(values.shape[1]), [COLS[c] for c in corr.columns])
    ax.set_yticks(range(values.shape[0]), [ROWS[r] for r in corr.index])
    ax.tick_params(length=0, labelcolor=INK)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=28)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corr-dir", type=Path, default=ROOT / "results" / "evaluation" / "group1_corr")
    p.add_argument("--n", type=int, default=39, help="number of logs, shown in the caption")
    args = p.parse_args()

    pearson = load(args.corr_dir, "p")
    kendall = load(args.corr_dir, "k")
    vlim = 0.7  # shared, symmetric scale so the two panels are comparable by color

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    draw(axes[0], *pearson, "Pearson r", vlim)
    im = draw(axes[1], *kendall, "Kendall τ", vlim)
    axes[1].tick_params(labelleft=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelcolor=INK_MUTED)
    cbar.set_label("correlation", color=INK_MUTED)

    fig.suptitle("Translucent complexity vs. F-score by discovery variant",
                 x=0.01, ha="left", fontsize=14, color=INK, y=1.12)
    fig.text(0.01, 0.04,
             f"n = {args.n} logs.  * p < 0.05   "
             "Blue: higher complexity, lower F-score.  τ is naturally smaller than r for the same strength.",
             fontsize=9, color=INK_MUTED, ha="left")

    for ext in ("png", "pdf"):
        out = args.corr_dir / f"complexity_fscore_heatmap.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        print("wrote", out)


if __name__ == "__main__":
    main()
