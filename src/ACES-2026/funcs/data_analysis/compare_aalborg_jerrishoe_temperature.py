"""
Vergleich der Lufttemperatur: Jerrishoe vs. Aalborg, Basisjahr 2019.
Datenquelle: ERA5-Reanalyse (Copernicus CDS) für beide Standorte, konsistent über
funcs/era5_weather.py -- ersetzt den früheren meteostat-Vergleich (Flensburg vs.
Aalborg, 2019-2021). Auf 2019 beschränkt (statt 2019-2021 wie zuvor), passend zum
für das Projekt festgelegten Basisjahr und zur PV-Vergleichsbasis (siehe
compare_aalborg_jerrishoe_pv.py, die ebenfalls nur 2019 nutzt).
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../funcs
from era5_weather import load_era5_weather, REPO_SRC_DIR

# Koordinaten
JERRISHOE = {"lat": 54.65699858112733, "lon": 9.37165074067953, "name": "Jerrishoe"}
AALBORG   = {"lat": 57.05, "lon": 9.92, "name": "Aalborg"}
YEAR      = 2019
CACHE_DIR = str(REPO_SRC_DIR / "Data" / "era5_cache")


def load_city(city: dict, year: int) -> pd.Series:
    df = load_era5_weather(year, lat=city["lat"], lon=city["lon"], cache_dir=CACHE_DIR)
    return df["T_amb_C"].rename(city["name"])


def print_stats(je: pd.Series, aa: pd.Series) -> None:
    print("\n" + "=" * 55)
    print(f"{'Kennwert':<28} {'Jerrishoe':>12} {'Aalborg':>12}")
    print("-" * 55)
    stats = {
        "Mittelwert [°C]":    (je.mean(),  aa.mean()),
        "Minimum [°C]":       (je.min(),   aa.min()),
        "Maximum [°C]":       (je.max(),   aa.max()),
        "Std.-Abw. [K]":      (je.std(),   aa.std()),
        "Stunden < 0 °C":     ((je < 0).sum(), (aa < 0).sum()),
        "Stunden < -5 °C":    ((je < -5).sum(), (aa < -5).sum()),
        "Heizgradtage (HGT)": (
            (12 - je.resample("D").mean()).clip(lower=0).sum(),
            (12 - aa.resample("D").mean()).clip(lower=0).sum(),
        ),
    }
    for label, (j_val, a_val) in stats.items():
        print(f"  {label:<26} {j_val:>12.1f} {a_val:>12.1f}")
    print("=" * 55 + "\n")


def plot_comparison(je: pd.Series, aa: pd.Series) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Temperaturvergleich Jerrishoe vs. Aalborg (ERA5, 2019)",
                 fontsize=14, fontweight="bold")

    # 1. Zeitreihe (7-Tage-Mittel)
    ax = axes[0, 0]
    ax.plot(je.resample("7D").mean(), label="Jerrishoe", color="#1565C0", lw=1.4)
    ax.plot(aa.resample("7D").mean(), label="Aalborg",   color="#C62828", lw=1.4)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Temperatur (7-Tage-Mittel)")
    ax.set_ylabel("Temperatur [°C]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Monatsmittelwerte
    ax = axes[0, 1]
    je_mon = je.groupby(je.index.month).mean()
    aa_mon = aa.groupby(aa.index.month).mean()
    months = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
    x = np.arange(12)
    w = 0.35
    ax.bar(x - w/2, je_mon.values, w, label="Jerrishoe", color="#1565C0", alpha=0.8)
    ax.bar(x + w/2, aa_mon.values, w, label="Aalborg",   color="#C62828", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_title(f"Monatsmitteltemperatur ({YEAR})")
    ax.set_ylabel("Temperatur [°C]")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Dauerlinie (sortierte Stundenwerte)
    ax = axes[1, 0]
    je_sorted = np.sort(je.dropna().values)[::-1]
    aa_sorted = np.sort(aa.dropna().values)[::-1]
    hours_je  = np.linspace(0, len(je_sorted), len(je_sorted))
    hours_aa  = np.linspace(0, len(aa_sorted), len(aa_sorted))
    ax.plot(hours_je, je_sorted, label="Jerrishoe", color="#1565C0", lw=1.4)
    ax.plot(hours_aa, aa_sorted, label="Aalborg",   color="#C62828", lw=1.4)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Temperaturdauerlinie")
    ax.set_xlabel("Stunden")
    ax.set_ylabel("Temperatur [°C]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Differenz Aalborg − Jerrishoe (Tagesmittel)
    ax = axes[1, 1]
    diff = aa.resample("D").mean() - je.resample("D").mean()
    diff = diff.dropna()
    ax.fill_between(diff.index, diff.values, 0,
                    where=(diff >= 0), color="#C62828", alpha=0.5, label="Aalborg wärmer")
    ax.fill_between(diff.index, diff.values, 0,
                    where=(diff < 0),  color="#1565C0", alpha=0.5, label="Jerrishoe wärmer")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Temperaturdifferenz Aalborg − Jerrishoe (Tagesmittel)")
    ax.set_ylabel("ΔT [K]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = str(REPO_SRC_DIR / "Data" / "temperature_comparison_je_aa.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out}")
    plt.show()


if __name__ == "__main__":
    print(f"Lade ERA5-Temperaturdaten ({YEAR}) …")
    print(f"  Jerrishoe ({JERRISHOE['lat']:.4f}°N, {JERRISHOE['lon']:.4f}°E)")
    je = load_city(JERRISHOE, YEAR)
    print(f"  Aalborg   ({AALBORG['lat']}°N, {AALBORG['lon']}°E)")
    aa = load_city(AALBORG, YEAR)

    print_stats(je, aa)
    plot_comparison(je, aa)
