"""Regenerate the README figures in docs/img/ from the sample data and a pipeline/1_train.py run.

    python pipeline/1_train.py           # writes outputs/
    python tools/make_readme_figures.py  # writes docs/img/tutorial_*.png

tutorial_ground_truth.png  five days of RIVN 5-min closes with the buy/sell labels of step [3]
tutorial_confusion.png     what the trained model predicted on the test split, per true label
tutorial_model.png         the network diagram that step [9] draws
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockpred.config import TIME_WINDOW_GT
from stockpred.ml.buy_sell_point import get_buy_sell_point_HT_pp
from stockpred.paths import MODEL_INFO, PLOTS, ROOT, SAMPLE

OUT = ROOT / "docs" / "img"
SYMBOL, REF_MODEL = "RIVN", "_3W5__tut_A1_"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BUY, SELL = "#2a78d6", "#e34948"          # validated pair (dataviz palette slots 1 and 8)
BLUES = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def ground_truth():
    df = pd.read_csv(SAMPLE / f"tutorial_pre_{SYMBOL}_5min.csv", parse_dates=["Date"], index_col="Date", sep="\t")
    df.columns = [c.lower() for c in df.columns]
    target, df_l, _ = get_buy_sell_point_HT_pp(df.copy(), period=TIME_WINDOW_GT, rolling=TIME_WINDOW_GT * 2)
    days = pd.Index(df_l.index.date).unique()
    win = df_l[pd.Index(df_l.index.date).isin(days[-12:-7])]       # five full days, away from the unlabeled tail
    x = np.arange(len(win))
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    ax.plot(x, win["close"], color=INK_2, linewidth=1.6, label="close (5 min)")
    buy, sell = win["target"].values == 1, win["target"].values == 2
    ax.scatter(x[buy], win["low"].values[buy], marker="^", s=70, color=BUY, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=f"buy label ({buy.sum()})")
    ax.scatter(x[sell], win["high"].values[sell], marker="v", s=70, color=SELL, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=f"sell label ({sell.sum()})")
    starts = [i for i in range(len(win)) if i == 0 or win.index[i].date() != win.index[i - 1].date()]
    ax.set_xticks(starts, [win.index[i].strftime("%a %d %b") for i in starts])
    for s in starts[1:]:
        ax.axvline(s - 0.5, color=GRID, linewidth=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("price ($)")
    ax.set_title(f"{SYMBOL}: the ground truth the model learns (step [3], window {TIME_WINDOW_GT} candles)", loc="left", fontsize=13)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "tutorial_ground_truth.png")
    print("wrote", OUT / "tutorial_ground_truth.png", f"({buy.sum()} buy, {sell.sum()} sell in {len(win)} candles)")


def confusion():
    df = pd.read_csv(MODEL_INFO / f"{SYMBOL}_{REF_MODEL}.csv", sep="\t", index_col=0)
    names = ["do nothing", "buy", "sell"]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(df["y_test"], df["y_pred"]):
        cm[t, p] += 1
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{share[i, j]:.0%}\n{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="#ffffff" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"Test split ({cm.sum():,} windows): share of each true label\naccuracy {acc:.0%}", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "tutorial_confusion.png")
    print("wrote", OUT / "tutorial_confusion.png", "rows:", cm.tolist())


def model_diagram():
    src = PLOTS / f"{SYMBOL}_{REF_MODEL}_buysell.png"
    shutil.copyfile(src, OUT / "tutorial_model.png")
    print("wrote", OUT / "tutorial_model.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ground_truth()
    confusion()
    model_diagram()
