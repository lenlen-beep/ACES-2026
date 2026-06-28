#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:48:09 2026

@author: nele
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "/Users/nele/Documents/GitHub/ACES-2026/src/ACES-2026/Data/Aalborg_smart_meter_data"
USECOLS  = ["CustomerID", "Energi 1 Varmeenergi", "Maks.-effekt 1",
            "Temperatur 1", "Temperatur 2", "RoundedReadTime"]

# -------------------------------------------------------------------------
# Kategorien: (Label, Jahresverbrauch_min kWh, Jahresverbrauch_max kWh)
# Grenzwerte typisch für dänische Fernwärme-Abnehmer.
# -------------------------------------------------------------------------
CATEGORIES = [
    ("Einfamilienhaus (EFH)",            0,       15_000),
    ("Kleines MFH / Reihenhaus",    15_000,       50_000),
    ("Großes MFH / Wohnblock",      50_000,      150_000),
    ("Gewerbe / Schule / Büro",    150_000,      500_000),
    ("Industrie / Großverbraucher", 500_000, float("inf")),
]

BUILDING_NOTES = {
    "Einfamilienhaus (EFH)":            "1–2 Wohneinheiten, Einfamilienhaus",
    "Kleines MFH / Reihenhaus":         "3–10 Wohneinheiten, Reihenhaus",
    "Großes MFH / Wohnblock":           "Mehrgeschossiger Wohnblock, 10–40 WE",
    "Gewerbe / Schule / Büro":          "Büro, Schule, Hotel, Supermarkt",
    "Industrie / Großverbraucher":      "Industriebetrieb oder Übergabestation",
}


def classify(annual_kwh: float) -> str:
    for label, lo, hi in CATEGORIES:
        if lo <= annual_kwh < hi:
            return label
    return "Unbekannt"


def load_all_files(data_dir: str = DATA_DIR) -> pd.DataFrame:
    all_csv = glob.glob(os.path.join(data_dir, "*.csv"))

    # Kontextdatei ausschließen
    files = [p for p in all_csv if os.path.basename(p) != "contextual_data.csv"]

    # numerische Sortierung nur für Dateien wie "1.csv", "2.csv", ...
    files = sorted(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    print(f"Lade {len(files)} Dateien …")
    chunks = []
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, usecols=USECOLS, low_memory=False)
        df["RoundedReadTime"] = pd.to_datetime(df["RoundedReadTime"],
                                               dayfirst=True, errors="coerce")
        df["Energi 1 Varmeenergi"] = pd.to_numeric(df["Energi 1 Varmeenergi"],
                                                    errors="coerce")
        df["Maks.-effekt 1"] = pd.to_numeric(df["Maks.-effekt 1"], errors="coerce")
        df["Temperatur 1"]   = pd.to_numeric(df["Temperatur 1"],   errors="coerce")
        df["Temperatur 2"]   = pd.to_numeric(df["Temperatur 2"],   errors="coerce")
        chunks.append(df)
        print(f"  {i:2d}/25  {os.path.basename(f)}  ({df['CustomerID'].nunique()} Kunden)")

    return pd.concat(chunks, ignore_index=True)


def build_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    def summarize(g):
        energy = g["Energi 1 Varmeenergi"].dropna()
        time   = g["RoundedReadTime"].dropna()
        span_years = (time.max() - time.min()).days / 365.25 if len(time) > 1 else np.nan
        total_kwh  = energy.max() - energy.min() if len(energy) > 1 else np.nan
        annual_kwh = total_kwh / span_years if span_years and span_years > 0 else np.nan
        return pd.Series({
            "Jahresverbrauch_kWh": round(annual_kwh) if not np.isnan(annual_kwh) else np.nan,
            "MaxLeistung_kW":      round(g["Maks.-effekt 1"].max(), 1),
            "T_VL_mittel_C":       round(g["Temperatur 1"].mean(), 1),
            "T_RL_mittel_C":       round(g["Temperatur 2"].mean(), 1),
            "Zeitraum_Jahre":      round(span_years, 2) if span_years else np.nan,
            "Anzahl_Messwerte":    len(g),
        })

    print("Berechne Kundenzusammenfassung …")
    summary = df.groupby("CustomerID").apply(summarize, include_groups=False).reset_index()
    summary = summary.dropna(subset=["Jahresverbrauch_kWh"])
    summary["Kategorie"] = summary["Jahresverbrauch_kWh"].apply(classify)
    summary = summary.sort_values("Jahresverbrauch_kWh").reset_index(drop=True)
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("AALBORG SMART METER — Gebäudekategorisierung")
    print("=" * 70)
    print(f"Gesamt analysierte Gebäude: {len(summary)}\n")

    print("Anzahl Gebäude je Kategorie:")
    print("-" * 70)
    counts = (summary["Kategorie"]
              .value_counts()
              .reindex([c[0] for c in CATEGORIES])
              .fillna(0)
              .astype(int))
    for cat, n in counts.items():
        note = BUILDING_NOTES.get(cat, "")
        print(f"  {n:4d}×  {cat:<35s}  {note}")

    print("\nStatistik Jahresverbrauch [kWh/Jahr]:")
    print(summary["Jahresverbrauch_kWh"].describe()
          .apply(lambda x: f"{x:,.0f}").to_string())
    print("=" * 70 + "\n")


def plot_results(summary: pd.DataFrame) -> None:
    cat_order  = [c[0] for c in CATEGORIES]
    cat_colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]
    color_map  = dict(zip(cat_order, cat_colors))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Aalborg Smart Meter — Gebäudekategorisierung",
                 fontsize=14, fontweight="bold")

    # 1. Histogramm Jahresverbrauch
    ax = axes[0]
    for cat, color in zip(cat_order, cat_colors):
        sub = summary[summary["Kategorie"] == cat]["Jahresverbrauch_kWh"] / 1000
        if len(sub):
            ax.hist(sub, bins=30, color=color, alpha=0.7, label=cat)
    ax.set_xlabel("Jahresverbrauch [MWh/Jahr]")
    ax.set_ylabel("Anzahl Gebäude")
    ax.set_title("Verteilung Jahresverbrauch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Scatter: Jahresverbrauch vs. Maximalleistung
    ax = axes[1]
    for cat in cat_order:
        grp = summary[summary["Kategorie"] == cat].dropna(subset=["MaxLeistung_kW"])
        if len(grp):
            ax.scatter(grp["Jahresverbrauch_kWh"] / 1000,
                       grp["MaxLeistung_kW"],
                       label=cat, color=color_map[cat],
                       s=10, alpha=0.6, zorder=3)
    ax.set_xlabel("Jahresverbrauch [MWh/Jahr]")
    ax.set_ylabel("Maximalleistung [kW]")
    ax.set_title("Verbrauch vs. Spitzenlast")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. Tortendiagramm
    ax = axes[2]
    counts = (summary["Kategorie"]
              .value_counts()
              .reindex(cat_order)
              .dropna()
              .astype(int))
    wedge_colors = [color_map[c] for c in counts.index]
    ax.pie(counts.values, labels=counts.index, colors=wedge_colors,
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title("Kategorieverteilung")

    plt.tight_layout()
    out = "//Users/nele/Documents/GitHub/ACES-2026/src/ACES-2026/Data/Aalborg_smart_meter_data/aalborg_kategorisierung.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out}")
    plt.show()


if __name__ == "__main__":
    df      = load_all_files()
    summary = build_customer_summary(df)
    print_summary(summary)
    plot_results(summary)
