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
import pandas as pd

ACES_DIR = Path(__file__).resolve().parent
RESULTS = ACES_DIR / "Data" / "sensitivity_results.csv"
PLOTS_DIR = ACES_DIR / "plots"

# Szenarienpaare: Anzeigename -> (Szenario bei -30 %, Szenario bei +30 %)
# Reihenfolge im Tornado ergibt sich aus der Spannweite, nicht aus dieser Liste.
# Szenarionummer -> (Anzeigename, Lauf mit niedrigem Wert, Lauf mit hohem Wert)
PAIRS = {
    1: ("Pipe investment cost",      "pipe_cost_minus30",    "pipe_cost_plus30"),
    2: ("Heat pump investment cost", "hp_cost_minus30",      "hp_cost_plus30"),
    3: ("Storage investment cost",   "storage_cost_minus30", "storage_cost_plus30"),
    4: ("Electricity market price",  "elec_price_minus30",   "elec_price_plus30"),
    5: ("Gas price",                 "gas_low",              "gas_high"),
    6: ("CO\u2082 price",            None,                   "co2_100"),
    7: ("Discount rate",             "interest_3pct",        "interest_7pct"),
}

LOW_COLOR = "#4C72B0"
HIGH_COLOR = "#C44E52"


def load():
    if not RESULTS.exists():
        raise SystemExit(f"FEHLER: {RESULTS} fehlt. Erst sensitivity.py laufen lassen.")
    df = pd.read_csv(RESULTS).set_index("scenario")
    if "base" not in df.index:
        raise SystemExit("FEHLER: Szenario 'base' fehlt in den Ergebnissen.")
    return df, float(df.loc["base", "lcoh_eur_per_mwh"])


def collect(df, base):
    """Liefert je Szenario (Name, LCOH_low, LCOH_high), sortiert nach Spannweite."""
    rows = []
    for no, (label, lo_key, hi_key) in PAIRS.items():
        lo = float(df.loc[lo_key, "lcoh_eur_per_mwh"]) if lo_key in df.index else base
        hi = float(df.loc[hi_key, "lcoh_eur_per_mwh"]) if hi_key in df.index else base
        if lo == base and hi == base:
            continue                     # Szenario nicht gerechnet -> auslassen
        rows.append((f"S{no}  {label}", lo, hi))
    rows.sort(key=lambda r: abs(r[2] - r[1]))     # kleinste Spannweite zuerst
    return rows


def tornado(rows, base):
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 2.2))
    y = range(len(rows))

    for i, (label, lo, hi) in enumerate(rows):
        ax.barh(i, lo - base, left=base, height=0.62,
                color=LOW_COLOR, edgecolor="white", zorder=3)
        ax.barh(i, hi - base, left=base, height=0.62,
                color=HIGH_COLOR, edgecolor="white", zorder=3)
        span = max(abs(hi - base), abs(lo - base))
        for val in (lo, hi):
            if abs(val - base) < 0.01 * base and span < 0.02 * base:
                continue                  # zu kleiner Balken -> Beschriftung wuerde kollidieren
            off = 2.5 if val >= base else -2.5
            ax.text(val + off, i, f"{val:.0f}", va="center",
                    ha="left" if val >= base else "right", fontsize=8.5, zorder=4)

    lo_all = min(min(r[1], r[2]) for r in rows)
    hi_all = max(max(r[1], r[2]) for r in rows)
    pad = 0.10 * (hi_all - lo_all)
    ax.set_xlim(lo_all - pad, hi_all + pad)
    ax.axvline(base, color="black", lw=1.2, zorder=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("LCOH (EUR/MWh)", fontsize=10)
    ax.set_title(f"Sensitivity of LCOH to input parameters "
                 f"(reference case: {base:.1f} EUR/MWh)", fontsize=11, pad=12)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=LOW_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=HIGH_COLOR)]
    ax.legend(handles, ["low variant", "high variant"],
              loc="lower right", fontsize=9, frameon=False)

    fig.tight_layout()
    out = PLOTS_DIR / "sensitivity_tornado.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Plot gespeichert: {out}")


def spider(rows, base):
    """Relative Aenderung des LCOH gegen relative Parameteraenderung."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [-30, 0, 30]

    for label, lo, hi in rows:
        ys = [(lo / base - 1) * 100, 0.0, (hi / base - 1) * 100]
        ax.plot(x, ys, marker="o", ms=5, lw=1.8, label=label)

    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(0, color="black", lw=0.9, alpha=0.4)
    ax.set_xlabel("Parameter variation (%)", fontsize=10)
    ax.set_ylabel("Change in LCOH (%)", fontsize=10)
    ax.set_title("Spider diagram: relative sensitivity of LCOH", fontsize=11, pad=12)
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    fig.tight_layout()
    out = PLOTS_DIR / "sensitivity_spider.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Plot gespeichert: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df, base = load()
    rows = collect(df, base)
    if not rows:
        raise SystemExit("Keine auswertbaren Szenarienpaare gefunden.")

    tornado(rows, base)
    spider(rows, base)

    print(f"\nReferenzfall: {base:.2f} EUR/MWh")
    print(f"{'Parameter':44s} {'low':>9s} {'high':>9s} {'Spannweite':>12s}")
    for label, lo, hi in reversed(rows):
        print(f"{label:44s} {lo:9.2f} {hi:9.2f} {abs(hi - lo):11.2f}")


if __name__ == "__main__":
    main()
