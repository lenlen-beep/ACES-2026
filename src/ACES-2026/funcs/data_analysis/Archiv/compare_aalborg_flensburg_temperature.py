import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

sys.path.insert(0, "src/ACES-2026/funcs")
from read_data import load_temperature_data

# Koordinaten
FLENSBURG = {"lat": 54.78, "lon": 9.43,  "name": "Flensburg"}
AALBORG   = {"lat": 57.05, "lon": 9.92,  "name": "Aalborg"}
YEARS     = [2019, 2020, 2021]
CACHE_DIR = "src/ACES-2026/Data/weather_cache"


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


def plot_comparison(fl: pd.Series, aa: pd.Series) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Temperaturvergleich Flensburg vs. Aalborg  (2019–2021)",
                 fontsize=14, fontweight="bold")

    # 1. Zeitreihe (7-Tage-Mittel)
    ax = axes[0, 0]
    ax.plot(fl.resample("7D").mean(), label="Flensburg", color="#1565C0", lw=1.4)
    ax.plot(aa.resample("7D").mean(), label="Aalborg",   color="#C62828", lw=1.4)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Temperatur (7-Tage-Mittel)")
    ax.set_ylabel("Temperatur [°C]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Monatsmittelwerte (alle Jahre gemittelt)
    ax = axes[0, 1]
    fl_mon = fl.groupby(fl.index.month).mean()
    aa_mon = aa.groupby(aa.index.month).mean()
    months = ["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]
    x = np.arange(12)
    w = 0.35
    ax.bar(x - w/2, fl_mon.values, w, label="Flensburg", color="#1565C0", alpha=0.8)
    ax.bar(x + w/2, aa_mon.values, w, label="Aalborg",   color="#C62828", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_title("Monatsmitteltemperatur (Ø 2019–2021)")
    ax.set_ylabel("Temperatur [°C]")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Dauerlinie (sortierte Stundenwerte)
    ax = axes[1, 0]
    fl_sorted = np.sort(fl.dropna().values)[::-1]
    aa_sorted = np.sort(aa.dropna().values)[::-1]
    hours_fl  = np.linspace(0, len(fl_sorted), len(fl_sorted))
    hours_aa  = np.linspace(0, len(aa_sorted), len(aa_sorted))
    ax.plot(hours_fl, fl_sorted, label="Flensburg", color="#1565C0", lw=1.4)
    ax.plot(hours_aa, aa_sorted, label="Aalborg",   color="#C62828", lw=1.4)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_title("Temperaturdauerlinie")
    ax.set_xlabel("Stunden")
    ax.set_ylabel("Temperatur [°C]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Differenz Aalborg − Flensburg (Tagesmittel)
    ax = axes[1, 1]
    diff = aa.resample("D").mean() - fl.resample("D").mean()
    diff = diff.dropna()
    ax.fill_between(diff.index, diff.values, 0,
                    where=(diff >= 0), color="#C62828", alpha=0.5, label="Aalborg wärmer")
    ax.fill_between(diff.index, diff.values, 0,
                    where=(diff < 0),  color="#1565C0", alpha=0.5, label="Flensburg wärmer")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Temperaturdifferenz Aalborg − Flensburg (Tagesmittel)")
    ax.set_ylabel("ΔT [K]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "src/ACES-2026/Data/temperature_comparison_fl_aa.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out}")
    plt.show()


if __name__ == "__main__":
    print("Lade Temperaturdaten …")
    print(f"  Flensburg ({FLENSBURG['lat']}°N, {FLENSBURG['lon']}°E)")
    fl = load_city(FLENSBURG, YEARS)
    print(f"  Aalborg   ({AALBORG['lat']}°N, {AALBORG['lon']}°E)")
    aa = load_city(AALBORG, YEARS)

    print_stats(fl, aa)
    plot_comparison(fl, aa)
