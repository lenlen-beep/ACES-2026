"""
Vergleich des PV-Ertrags: Jerrishoe vs. Aalborg, 2019.
Datenquelle: ERA5-Strahlungsdaten (Copernicus CDS) + eigenes PV-Modell (pvlib),
konsistent über funcs/era5_weather.py für beide Standorte -- ersetzt den früheren
renewables.ninja-Vergleich (Flensburg vs. Aalborg, MERRA-2-Datensatz).

Anlagenkonfiguration (normiert auf 1 kWp, wie zuvor bei renewables.ninja):
  Neigung 35°, Azimut 180° (Süd), Performance Ratio 0.85 (System_parameters.PV
  in parameters.yaml).
"""

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../funcs
from era5_weather import load_era5_weather, compute_pv_generation, REPO_SRC_DIR

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

JERRISHOE = {"lat": 54.65699858112733, "lon": 9.37165074067953, "name": "Jerrishoe"}
AALBORG   = {"lat": 57.05, "lon": 9.92, "name": "Aalborg"}
YEAR      = 2019
CACHE_DIR = str(REPO_SRC_DIR / "Data" / "era5_cache")
OUT_DIR   = str(REPO_SRC_DIR / "plots")

SURFACE_TILT     = 35
SURFACE_AZIMUTH  = 180
PV_CAPACITY_KWP  = 1.0                 # normiert, wie beim alten ninja-Vergleich
PERFORMANCE_RATIO = 0.85

C_JERRISHOE = "#A0463A"   # Gedämpftes Rot
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


def load_pv(city: dict, name: str) -> pd.Series:
    weather = load_era5_weather(YEAR, lat=city["lat"], lon=city["lon"], cache_dir=CACHE_DIR)
    # PV_CAPACITY_KWP = 1 kWp -> pv_capacity_MW = 1e-3, Ergebnis direkt in kW/kWp lesbar
    pv_mw = compute_pv_generation(
        weather, lat=city["lat"], lon=city["lon"],
        surface_tilt=SURFACE_TILT, surface_azimuth=SURFACE_AZIMUTH,
        pv_capacity_MW=PV_CAPACITY_KWP / 1000, performance_ratio=PERFORMANCE_RATIO,
    )
    return (pv_mw * 1000).rename(name)  # MW -> kW (bei 1 kWp: kW == kW/kWp)


def print_stats(je: pd.Series, aa: pd.Series) -> None:
    je_kwh = je.sum()
    aa_kwh = aa.sum()
    print("\n" + "=" * 55)
    print(f"{'Kennwert':<30} {'Jerrishoe':>10} {'Aalborg':>10}")
    print("-" * 55)
    stats = {
        "Jahresertrag [kWh/kWp]": (je_kwh, aa_kwh),
        "Max. Leistung [kW/kWp]": (je.max(), aa.max()),
        "Volllaststunden [h/a]":  (je_kwh, aa_kwh),  # identisch bei normiertem 1 kWp
        "Stunden > 0.5 kW/kWp":  ((je > 0.5).sum(), (aa > 0.5).sum()),
        "Stunden > 0 [h/a]":     ((je > 0).sum(),   (aa > 0).sum()),
    }
    for label, (j_val, a_val) in stats.items():
        print(f"  {label:<28} {j_val:>10.1f} {a_val:>10.1f}")
    print("=" * 55 + "\n")


def plot_main(je: pd.Series, aa: pd.Series) -> None:
    """Monthly mean (bar, left) + load duration curve (right)."""
    je_mon = je.groupby(je.index.month).mean()
    aa_mon = aa.groupby(aa.index.month).mean()

    je_sorted = np.sort(je.values)[::-1]
    aa_sorted = np.sort(aa.values)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(wspace=0.13, left=0.07, right=0.97, top=0.88, bottom=0.12)
    fig.suptitle("PV yield comparison – Jerrishoe vs. Aalborg (ERA5, 2019)",
                 fontsize=TITLE_FS, fontweight="bold", color="#1A1A1A")

    # --- links: Monatsmittel ---
    x = np.arange(12)
    w = 0.38
    ax1.bar(x - w / 2, je_mon.values, w, color=C_JERRISHOE, alpha=0.85, label="Jerrishoe")
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
    ax2.plot(je_sorted, color=C_JERRISHOE, linewidth=1.6, label="Jerrishoe")
    ax2.plot(aa_sorted, color=C_AALBORG,   linewidth=1.6, label="Aalborg")
    ax2.set_xlabel("Hours (sorted)", fontsize=LABEL_FS)
    ax2.set_ylabel("PV output in kW/kWp", fontsize=LABEL_FS)
    ax2.set_title("PV load duration curve", fontsize=LABEL_FS)
    ax2.legend(fontsize=LEGEND_FS, frameon=False)
    _ppt_style(ax2)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "pv_comparison_jerrishoe_aalborg.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    plt.show()
    plt.close(fig)


def plot_difference(je: pd.Series, aa: pd.Series) -> None:
    """Monatliche Differenz Aalborg − Jerrishoe."""
    je_mon = je.groupby(je.index.month).sum()
    aa_mon = aa.groupby(aa.index.month).sum()
    diff   = aa_mon - je_mon

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12)
    fig.suptitle("PV yield difference Aalborg − Jerrishoe (monthly total, 2019)",
                 fontsize=TITLE_FS, fontweight="bold", color="#1A1A1A")

    colors = [C_AALBORG if v >= 0 else C_JERRISHOE for v in diff.values]
    ax.bar(np.arange(12), diff.values, color=colors, alpha=0.85)
    ax.axhline(0, color="#1A1A1A", linewidth=0.9)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(MONTHS, fontsize=TICK_FS)
    ax.set_ylabel("ΔPV output in kWh/kWp", fontsize=LABEL_FS)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=C_AALBORG,   alpha=0.85, label="Aalborg higher"),
                       Patch(facecolor=C_JERRISHOE, alpha=0.85, label="Jerrishoe higher")]
    ax.legend(handles=legend_elements, fontsize=LEGEND_FS, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, color="#CCCCCC", axis="y")
    ax.tick_params(labelsize=TICK_FS)
    ax.margins(x=0.04)

    out = os.path.join(OUT_DIR, "pv_difference_jerrishoe_aalborg.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    print(f"Lade ERA5-PV-Daten ({YEAR}) …")
    je = load_pv(JERRISHOE, "Jerrishoe")
    aa = load_pv(AALBORG, "Aalborg")

    print_stats(je, aa)
    plot_main(je, aa)
    plot_difference(je, aa)
