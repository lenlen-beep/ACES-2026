import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

sys.path.insert(0, "src/ACES-2026/funcs")
from read_data import load_temperature_data

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

FLENSBURG = {"lat": 54.78, "lon": 9.43,  "name": "Flensburg"}
AALBORG   = {"lat": 57.05, "lon": 9.92,  "name": "Aalborg"}
YEARS     = [2019, 2020, 2021]
CACHE_DIR = "src/ACES-2026/Data/weather_cache"
OUT_DIR   = "src/ACES-2026/plots"

C_AALBORG   = "#00395B"   # EUF-Blau
C_FLENSBURG = "#A0463A"   # Gedämpftes Rot

LABEL_FS  = 15
TICK_FS   = 13
LEGEND_FS = 13
TITLE_FS  = 16


def _ppt_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, color="#CCCCCC")
    ax.tick_params(labelsize=TICK_FS)
    ax.margins(x=0)


def load_city(city: dict, years: list) -> pd.Series:
    parts = []
    for y in years:
        s = load_temperature_data(y, lat=city["lat"], lon=city["lon"], cache_dir=CACHE_DIR)
        parts.append(s)
    return pd.concat(parts).sort_index().rename(city["name"])


def print_stats(fl: pd.Series, aa: pd.Series) -> None:
    print("\n" + "=" * 55)
    print(f"{'Kennwert':<28} {'Flensburg':>12} {'Aalborg':>12}")
    print("-" * 55)
    stats = {
        "Mittelwert [°C]":    (fl.mean(),  aa.mean()),
        "Minimum [°C]":       (fl.min(),   aa.min()),
        "Maximum [°C]":       (fl.max(),   aa.max()),
        "Std.-Abw. [K]":      (fl.std(),   aa.std()),
        "Stunden < 0 °C":     ((fl < 0).sum(), (aa < 0).sum()),
        "Stunden < -5 °C":    ((fl < -5).sum(), (aa < -5).sum()),
        "Heizgradtage (HGT)": (
            (12 - fl.resample("D").mean()).clip(lower=0).sum(),
            (12 - aa.resample("D").mean()).clip(lower=0).sum(),
        ),
    }
    for label, (f_val, a_val) in stats.items():
        print(f"  {label:<26} {f_val:>12.1f} {a_val:>12.1f}")
    print("=" * 55 + "\n")


def plot_main(fl: pd.Series, aa: pd.Series) -> None:
    """7-day mean (left) + load duration curve (right) side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(wspace=0.12, left=0.07, right=0.97, top=0.88, bottom=0.12)

    fig.suptitle("Temperature comparison – Flensburg vs. Aalborg (2019–2021)",
                 fontsize=TITLE_FS, fontweight="bold", color="#1A1A1A")

    # --- links: 7-Tage-Mittel ---
    ax1.plot(fl.resample("7D").mean(), color=C_FLENSBURG, linewidth=1.6, label="Flensburg")
    ax1.plot(aa.resample("7D").mean(), color=C_AALBORG,   linewidth=1.6, label="Aalborg")
    ax1.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Temperature in °C", fontsize=LABEL_FS)
    ax1.set_title("7-day moving average", fontsize=LABEL_FS)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=TICK_FS)
    ax1.legend(fontsize=LEGEND_FS, frameon=False)
    _ppt_style(ax1)

    # --- rechts: Dauerlinie ---
    fl_sorted = np.sort(fl.dropna().values)[::-1]
    aa_sorted = np.sort(aa.dropna().values)[::-1]
    ax2.plot(fl_sorted, color=C_FLENSBURG, linewidth=1.6, label="Flensburg")
    ax2.plot(aa_sorted, color=C_AALBORG,   linewidth=1.6, label="Aalborg")
    ax2.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Hours (sorted)", fontsize=LABEL_FS)
    ax2.set_ylabel("Temperature in °C", fontsize=LABEL_FS)
    ax2.set_title("Temperature duration curve", fontsize=LABEL_FS)
    ax2.legend(fontsize=LEGEND_FS, frameon=False)
    _ppt_style(ax2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "temperature_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    plt.show()
    plt.close(fig)


def plot_difference(fl: pd.Series, aa: pd.Series) -> None:
    """Tägliche Temperaturdifferenz Aalborg − Flensburg."""
    diff = aa.resample("D").mean() - fl.resample("D").mean()
    diff = diff.dropna()

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12)

    fig.suptitle("Temperature difference Aalborg − Flensburg (daily mean, 2019–2021)",
                 fontsize=TITLE_FS, fontweight="bold", color="#1A1A1A")

    ax.fill_between(diff.index, diff.values, 0,
                    where=(diff >= 0), color=C_AALBORG,   alpha=0.5, label="Aalborg warmer")
    ax.fill_between(diff.index, diff.values, 0,
                    where=(diff <  0), color=C_FLENSBURG, alpha=0.5, label="Flensburg warmer")
    ax.axhline(0, color="#1A1A1A", linewidth=0.8)
    ax.set_ylabel("ΔT in K", fontsize=LABEL_FS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    _ppt_style(ax)

    out = os.path.join(OUT_DIR, "temperature_difference.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    print("Lade Temperaturdaten …")
    fl = load_city(FLENSBURG, YEARS)
    aa = load_city(AALBORG,   YEARS)

    print_stats(fl, aa)
    plot_main(fl, aa)
    plot_difference(fl, aa)
