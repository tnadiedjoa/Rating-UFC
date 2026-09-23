"""
Shared matplotlib style for the notebooks.

Colours come from a colour-blind-checked categorical palette: use the slots
in order (BLUE, ORANGE, AQUA) and fold anything beyond three series into
grey or small multiples.
"""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9893"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]


def setup() -> None:
    """Apply the project style to every following figure."""
    plt.rcParams.update({
        "figure.figsize": (10, 4.2),
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "text.color": INK,
    })


def ranking_backtest_chart(table, title="How often the better-ranked fighter won"):
    """
    Grouped bars from ranking.backtest.accuracy_by_period(): for each period,
    the share of fights won by the favourite of each predictor. The betting
    market is drawn in grey, as a reference rather than a competitor.
    """
    import numpy as np
    import matplotlib.ticker as mtick

    predictors = [("Official rankings", "UFC official rankings", ORANGE),
                  ("Elo", "Elo rating", AQUA),
                  ("Model", "Model ranking", BLUE),
                  ("Betting favourite", "Betting favourite (reference)", NEUTRAL)]
    x = np.arange(len(table))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (column, label, color) in enumerate(predictors):
        values = table[column].to_numpy()
        bars = ax.bar(x + (i - 1.5) * width, values, width, color=color, label=label,
                      edgecolor=SURFACE, linewidth=2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.0%}",
                    ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    ax.axhline(0.5, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set_xlim(-0.5, x[-1] + 0.9)   # room for the label of the coin-flip line
    ax.text(x[-1] + 0.5, 0.5, " coin flip", va="center", ha="left", fontsize=8, color=INK_SECONDARY,
            backgroundcolor=SURFACE)
    ax.axvline(x[-1] - 0.5, color=GRID, linewidth=1.5)   # the last group sums up the others
    ax.set_xticks(x, [f"{period}\n{int(n):,} fights" for period, n in zip(table.index, table["fights"])])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_title(title, pad=26)
    ax.text(0, 1.03, "Fights between two ranked UFC fighters; each ranking as it stood the day before the event",
            transform=ax.transAxes, fontsize=9, color=INK_SECONDARY)
    ax.legend(ncols=4, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.tight_layout()
    return fig, ax
