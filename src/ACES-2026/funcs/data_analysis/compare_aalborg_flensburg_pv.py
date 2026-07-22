"""
Comparison of PV yield data: Flensburg vs. Aalborg
Data source: renewables.ninja (https://www.renewables.ninja)
  - Capacity: 1 kWp, tilt: 35°, azimuth: 180° (south), system loss: 10 %
  - Dataset: MERRA-2, year: 2019

Aalborg file must be downloaded manually from renewables.ninja:
  lat=57.0500, lon=9.9200, same parameters as Flensburg
  Save as: src/ACES-2026/Data/ninja_pv_57.0500_9.9200_corrected.csv
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src/ACES-2026/funcs")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

FILE_FL = "src/ACES-2026/Data/Solar_data/ninja_pv_54.7833_9.4333_corrected.csv"
FILE_AA = "src/ACES-2026/Data/Solar_data/ninja_pv_57.0444_9.9289_corrected.csv"
OUT_DIR = "src/ACES-2026/plots"

C_FLENSBURG = "#A0463A"   # Gedämpftes Rot
C_AALBORG   = "#00395B"   # EUF-Blau

LABEL_FS  = 15
TICK_FS   = 13
LEGEND_FS = 13
TITLE_FS  = 16

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _ppt_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, color="#CCCCCC")
    ax.tick_params(labelsize=TICK_FS)
    ax.margins(x=0)


def load_pv(filepath: str, name: str) -> pd.Series:
    df = pd.read_csv(filepath, header=3)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")[["electricity"]]
    df.index = df.index.tz_localize(None)
    return df["electricity"].rename(name)


def print_stats(fl: pd.Series, aa: pd.Series) -> None:
    fl_kwh = fl.sum()
    aa_kwh = aa.sum()
    print("\n" + "=" * 55)
    print(f"{'Kennwert':<30} {'Flensburg':>10} {'Aalborg':>10}")
    print("-" * 55)
    stats = {
        "Jahresertrag [kWh/kWp]": (fl_kwh, aa_kwh),
        "Max. Leistung [kW/kWp]": (fl.max(), aa.max()),
        "Volllaststunden [h/a]":  (fl_kwh, aa_kwh),  # identical for normalised 1 kWp
        "Stunden > 0.5 kW/kWp":  ((fl > 0.5).sum(), (aa > 0.5).sum()),
        "Stunden > 0 [h/a]":     ((fl > 0).sum(),   (aa > 0).sum()),
    }
    for label, (f_val, a_val) in stats.items():
        print(f"  {label:<28} {f_val:>10.1f} {a_val:>10.1f}")
    print("=" * 55 + "\n")


def plot_main(fl: pd.Series, aa: pd.Series) -> None:
    """Monthly mean (bar, left) + load duration curve (right)."""
    fl_mon = fl.groupby(fl.index.month).mean()
    aa_mon = aa.groupby(aa.index.month).mean()

    fl_sorted = np.sort(fl.values)[::-1]
    aa_sorted = np.sort(aa.values)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(wspace=0.13, left=0.07, right=0.97, top=0.88, bottom=0.12)
    fig.suptitle("PV yield comparison – Flensburg vs. Aalborg (2019)",
                 fontsize=TITLE_FS, fontweight="bold", color="#1A1A1A")

    # --- links: Monatsmittel ---
    x = np.arange(12)
    w = 0.38
    ax1.bar(x - w / 2, fl_mon.values, w, color=C_FLENSBURG, alpha=0.85, label="Flensburg")
    ax1.bar(x + w / 2, aa_mon.values, w, color=C_AALBORG,   alpha=0.85, label="Aalborg")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MONTHS, fontsize=TICK_FS)
    ax1.set_ylabel("Mean PV output in kW/kWp", fontsize=LABEL_FS)
    ax1.set_title("Monthly mean PV output", fontsize=LABEL_FS)
    ax1.legend(fontsize=LEGEND_FS, frameon=False)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(True, alpha=0.2, color="#CCCCCC", axis="y")
    ax1.tick_params(labelsize=TICK_FS)
    ax1.margins(x=0.04)

    # --- rechts: Dauerlinie ---
    ax2.plot(fl_sorted, color=C_FLENSBURG, linewidth=1.6, label="Flensburg")
    ax2.plot(aa_sorted, color=C_AALBORG,   linewidth=1.6, label="Aalborg")
    ax2.set_xlabel("Hours (sorted)", fontsize=LABEL_FS)
    ax2.set_ylabel("PV output in kW/kWp", fontsize=LABEL_FS)
    ax2.set_title("PV load duration curve", fontsize=LABEL_FS)
    ax2.legend(fontsize=LEGEND_FS, frameon=False)
    _ppt_style(ax2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "pv_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    plt.show()
    plt.close(fig)


def plot_difference(fl: pd.Series, aa: pd.Series) -> None:
    """Monatliche Differenz Aalborg − Flensburg."""
    fl_mon = fl.groupby(fl.index.month).sum()
    aa_mon = aa.groupby(aa.index.month).sum()
    diff   = aa_mon - fl_mon

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12)
    fig.suptitle("PV yield difference Aalborg − Flensburg (monthly total, 2019)",
                 fontsize=TITLE_FS, fontweight="bold", color="#1A1A1A")

    colors = [C_AALBORG if v >= 0 else C_FLENSBURG for v in diff.values]
    ax.bar(np.arange(12), diff.values, color=colors, alpha=0.85)
    ax.axhline(0, color="#1A1A1A", linewidth=0.9)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTHS, fontsize=TICK_FS)
    ax.set_ylabel("ΔPV output in kWh/kWp", fontsize=LABEL_FS)

    # Legende manuell
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=C_AALBORG,   alpha=0.85, label="Aalborg higher"),
                       Patch(facecolor=C_FLENSBURG, alpha=0.85, label="Flensburg higher")]
    ax.legend(handles=legend_elements, fontsize=LEGEND_FS, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, color="#CCCCCC", axis="y")
    ax.tick_params(labelsize=TICK_FS)
    ax.margins(x=0.04)

    out = os.path.join(OUT_DIR, "pv_difference.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if not os.path.exists(FILE_AA):
        print(f"\n⚠  Aalborg-Datei fehlt: {FILE_AA}")
        print("   Bitte von renewables.ninja herunterladen:")
        print("   lat=57.05, lon=9.92, tilt=35°, azimuth=180°, capacity=1 kWp, Jahr=2019\n")
        sys.exit(1)

    print("Lade PV-Daten …")
    fl = load_pv(FILE_FL, "Flensburg")
    aa = load_pv(FILE_AA, "Aalborg")

    print_stats(fl, aa)
    plot_main(fl, aa)
    plot_difference(fl, aa)
