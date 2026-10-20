"""
Tornado- und Spider-Diagramm zur Sensitivitaetsanalyse.

Aufruf aus dem Repo-Wurzelverzeichnis, nachdem sensitivity.py gelaufen ist:

    python src/ACES-2026/plot_sensitivity.py

Liest Data/sensitivity_results.csv und schreibt nach plots/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# --- Stil identisch zu funcs/plots.py ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

LABEL_FONTSIZE  = 14
TICK_FONTSIZE   = 14
LEGEND_FONTSIZE = 14
TITLE_FONTSIZE  = 14
VALUE_FONTSIZE  = 12

# Farben: niedrige Variante = EUF Blue (Wärmepumpe), hohe = Amber (Gas/fossil)
LOW_COLOR  = "#00395B"   # EUF blue
HIGH_COLOR = "#C17A2F"   # Amber

ACES_DIR  = Path(__file__).resolve().parent
RESULTS   = ACES_DIR / "Data" / "sensitivity_results.csv"
PLOTS_DIR = ACES_DIR / "plots"

PAIRS = {
    1: ("Pipe investment cost",       "pipe_cost_minus30",    "pipe_cost_plus30"),
    2: ("Heat pump investment cost",  "hp_cost_minus30",      "hp_cost_plus30"),
    3: ("Storage investment cost",    "storage_cost_minus30", "storage_cost_plus30"),
    4: ("Electricity market price",   "elec_price_minus30",   "elec_price_plus30"),
    5: ("Gas price",                  "gas_low",              "gas_high"),
    6: ("CO\u2082 price",            None,                   "co2_100"),
    7: ("Discount rate",              "interest_3pct",        "interest_7pct"),
}


def load():
    if not RESULTS.exists():
        raise SystemExit(f"ERROR: {RESULTS} not found. Run sensitivity.py first.")
    df = pd.read_csv(RESULTS).set_index("scenario")
    if "base" not in df.index:
        raise SystemExit("ERROR: scenario 'base' missing from results.")
    return df, float(df.loc["base", "lcoh_eur_per_mwh"])


def collect(df, base):
    rows = []
    for no, (label, lo_key, hi_key) in PAIRS.items():
        lo = float(df.loc[lo_key, "lcoh_eur_per_mwh"]) if lo_key in df.index else base
        hi = float(df.loc[hi_key, "lcoh_eur_per_mwh"]) if hi_key in df.index else base
        if lo == base and hi == base:
            continue
        rows.append((f"S{no}  {label}", lo, hi))
    rows.sort(key=lambda r: abs(r[2] - r[1]))
    return rows


def tornado(rows, base):
    fig, ax = plt.subplots(figsize=(10, 0.65 * len(rows) + 2.4), facecolor="white")
    y = range(len(rows))

    for i, (label, lo, hi) in enumerate(rows):
        ax.barh(i, lo - base, left=base, height=0.58,
                color=LOW_COLOR, edgecolor="white", zorder=3)
        ax.barh(i, hi - base, left=base, height=0.58,
                color=HIGH_COLOR, edgecolor="white", zorder=3)
        span = max(abs(hi - base), abs(lo - base))
        for val in (lo, hi):
            if abs(val - base) < 0.01 * base and span < 0.02 * base:
                continue
            off = 2.0 if val >= base else -2.0
            ax.text(val + off, i, f"{val:.0f}", va="center",
                    ha="left" if val >= base else "right",
                    fontsize=VALUE_FONTSIZE, zorder=4)

    lo_all = min(min(r[1], r[2]) for r in rows)
    hi_all = max(max(r[1], r[2]) for r in rows)
    pad = 0.10 * (hi_all - lo_all)
    ax.set_xlim(lo_all - pad, hi_all + pad)
    ax.axvline(base, color="#1A1A1A", lw=1.2, zorder=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=TICK_FONTSIZE)
    ax.set_xlabel("LCOH (EUR/MWh)", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        f"Sensitivity of LCOH to input parameters "
        f"(reference: {base:.1f} EUR/MWh)",
        fontsize=TITLE_FONTSIZE, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.2, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    handles = [plt.Rectangle((0, 0), 1, 1, color=LOW_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=HIGH_COLOR)]
    ax.legend(handles, ["low variant", "high variant"],
              loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=False)

    fig.tight_layout()
    out = PLOTS_DIR / "sensitivity_tornado.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved: {out}")


def spider(rows, base):
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="white")
    x = [-30, 0, 30]

    # Farben passend zu plots.py
    COLORS = [
        "#00395B",  # EUF blue
        "#C17A2F",  # Amber
        "#769D7B",  # Mint
        "#2F6B4F",  # Dark green
        "#A0463A",  # Muted red
        "#C8A84B",  # Gold
        "#1A1A1A",  # Near-black
    ]

    for idx, (label, lo, hi) in enumerate(rows):
        ys = [(lo / base - 1) * 100, 0.0, (hi / base - 1) * 100]
        ax.plot(x, ys, marker="o", ms=6, lw=1.8,
                color=COLORS[idx % len(COLORS)], label=label)

    ax.axhline(0, color="#1A1A1A", lw=0.9)
    ax.axvline(0, color="#1A1A1A", lw=0.9, alpha=0.4)
    ax.set_xlabel("Parameter variation (%)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Change in LCOH (%)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Sensitivity of LCOH — spider diagram",
                 fontsize=TITLE_FONTSIZE, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.2, color="#CCCCCC")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")

    fig.tight_layout()
    out = PLOTS_DIR / "sensitivity_spider.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df, base = load()
    rows = collect(df, base)
    if not rows:
        raise SystemExit("No scenario pairs found.")

    tornado(rows, base)
    spider(rows, base)

    print(f"\nReference case: {base:.2f} EUR/MWh")
    print(f"{'Parameter':44s} {'low':>9s} {'high':>9s} {'Spread':>12s}")
    for label, lo, hi in reversed(rows):
        print(f"{label:44s} {lo:9.2f} {hi:9.2f} {abs(hi - lo):11.2f}")


if __name__ == "__main__":
    main()