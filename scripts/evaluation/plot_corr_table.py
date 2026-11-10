"""
Paper-style correlation table:
    left   correlations among the complexity metrics (lower triangle)
    right  correlations between the complexity metrics and model quality (F-score) per discovery variant

Each cell shows Pearson r (top, black) and Kendall τ (bottom, gray), with stars for significance:
    *  p < 0.05     ** p < 0.01
Cell shading: dark gray = significant (p < 0.05) in both coefficients, light gray = in one of them.

Reads <corr-dir>/{p,k}_corr.csv and {p,k}_p_values.csv (square matrices saved without row labels,
rows in the same order as the columns) and writes complexity_fscore_table.png / .pdf next to them.

Usage (from the repo root):
    venv\\Scripts\\python.exe scripts\\evaluation\\plot_corr_table.py
    venv\\Scripts\\python.exe scripts\\evaluation\\plot_corr_table.py --corr-dir results\\evaluation\\group1_corr --title "Table 5: ..."
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

ROWS = {
    "complexity_avg_raw": "avg-raw-c",
    "complexity_avg_norm": "avg-norm-c",
    "complexity_div": "div-c",
    "complexity_disp": "disp-c",
    "complexity_diss_aa_seq": "diss-c",
}
LEFT_COLS = list(ROWS)[:-1]  # lower triangle: the last metric never appears as a column
RIGHT_COLS = {
    "f_score_imts": "IMts",
    "f_score_imto": "IMto",
    "f_score_imtf": "IMtf",
    "f_score_imfts": "IMfts",
    "f_score_imfto": "IMfto",
    "f_score_imftf": "IMftf",
}
ALPHA = 0.05

FILL = {1: "#e4e4e4", 2: "#c4c4c4"}  # significant in one / in both
INK, INK_SECOND = "#000000", "#6b6b6b"

# layout in data units
LABEL_W, LEFT_W, GAP_W, RIGHT_W = 1.25, 1.3, 0.25, 1.05
NUM_RIGHT = 0.22  # numbers are right-aligned this far right of the cell center, stars follow
ROW_H, HEADER_H = 1.0, 0.6


def load(corr_dir: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = pd.read_csv(corr_dir / f"{prefix}_corr.csv")
    pval = pd.read_csv(corr_dir / f"{prefix}_p_values.csv")
    corr.index, pval.index = corr.columns, pval.columns  # saved without row labels
    return corr, pval


def stars(p: float) -> str:
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def put_value(ax, x_center: float, y: float, v: float, p: float, color: str):
    """Number right-aligned at a fixed point (so decimals line up), stars raised just after it."""
    x = x_center + NUM_RIGHT
    ax.text(x, y, f"{v:.3f}".replace("-", "−"), ha="right", va="center", fontsize=11, color=color)
    if stars(p):
        ax.text(x + 0.01, y - 0.07, stars(p), ha="left", va="center", fontsize=9, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr-dir", type=Path, default=ROOT / "results" / "evaluation" / "group1_corr")
    ap.add_argument("--title", default="Correlations among Complexity Metrics (left) and between "
                                       "Complexity Metrics and Model Quality F1 (right)")
    args = ap.parse_args()

    p_corr, p_pval = load(args.corr_dir, "p")
    k_corr, k_pval = load(args.corr_dir, "k")

    # column x-positions (left edge, width)
    cols = []
    x = LABEL_W
    for c in LEFT_COLS:
        cols.append((c, ROWS[c], x, LEFT_W))
        x += LEFT_W
    sep_x = x + GAP_W / 2
    x += GAP_W
    for c, label in RIGHT_COLS.items():
        cols.append((c, label, x, RIGHT_W))
        x += RIGHT_W
    total_w = x
    total_h = HEADER_H + ROW_H * len(ROWS)

    fig, ax = plt.subplots(figsize=(total_w * 0.95, total_h * 0.62))
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    # header
    for _, label, x0, w in cols:
        ax.text(x0 + w / 2, HEADER_H / 2, label, ha="center", va="center", fontsize=11, fontweight="bold")

    for i, (row, row_label) in enumerate(ROWS.items()):
        y0 = HEADER_H + i * ROW_H
        ax.text(0.05, y0 + ROW_H / 2, row_label, ha="left", va="center", fontsize=11, fontweight="bold")
        for j, (col, _, x0, w) in enumerate(cols):
            if j < len(LEFT_COLS) and j >= i:  # left block: lower triangle only
                continue
            pr, pk = p_pval.at[row, col], k_pval.at[row, col]
            level = int(pr < ALPHA) + int(pk < ALPHA)
            if level:
                ax.add_patch(Rectangle((x0 + 0.02, y0 + 0.04), w - 0.04, ROW_H - 0.08,
                                       facecolor=FILL[level], edgecolor="none"))
            put_value(ax, x0 + w / 2, y0 + 0.32 * ROW_H, p_corr.at[row, col], pr, INK)
            put_value(ax, x0 + w / 2, y0 + 0.70 * ROW_H, k_corr.at[row, col], pk, INK_SECOND)

    # rules: top, below header, bottom; thick vertical separator between the two blocks
    for y, lw in ((0, 1.6), (HEADER_H, 0.8), (total_h, 1.6)):
        ax.plot([0, total_w], [y, y], color=INK, linewidth=lw, solid_capstyle="butt")
    ax.plot([sep_x, sep_x], [0.08, total_h - 0.08], color=INK, linewidth=2.4)

    ax.set_title(args.title, loc="left", fontsize=12, pad=8)
    ax.text(0, total_h + 0.3,
            "Top: Pearson r; bottom: Kendall τ.  * p < 0.05, ** p < 0.01.  "
            "Dark gray: significant in both; light gray: significant in one.",
            fontsize=9, color=INK_SECOND, ha="left", va="center", clip_on=False)

    for ext in ("png", "pdf"):
        out = args.corr_dir / f"complexity_fscore_table.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print("wrote", out)


if __name__ == "__main__":
    main()
