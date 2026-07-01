@ -1,84 +1,115 @@
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACES Projekt 2026 – Step 1 (methodisch korrigiert)

Ziel:
- Smart-Meter-CSV (exakt 25 Rohdaten-Dateien) + contextual_data.csv einlesen
- Für 2019 (stündlich):
  - Wärmeleistung Effekt 1 (kW) je Meter -> stündliche Matrix (8760 x n_meter), NaNs bleiben NaNs
  - Jahresverbrauch je Meter aus Energi 1 Varmeenergi (kWh): max-min innerhalb 2019
  - Spitzenlast je Meter: tatsächlicher Peak direkt aus Effekt 1 (stündliche Maxima)
- Plots:
  1) Histogramm Jahresverbrauch (MWh/Jahr) nach unit_type
  2) Scatter Jahresverbrauch vs. tatsächliche Spitzenlast (aus Effekt 1) nach unit_type
  3) Pie: Anzahl Meter je unit_type (Haustyp-Struktur des Datensatzes)
  4) Gesamtwärmeleistung (nansum Effekt 1) über 2019 + NaN-Anteil als Hilfslinie
  5) Gesamtwärmeleistung nach unit_type (Stack, nansum) + Pie nach Jahresenergie (Effekt 1)
  6) Gesamtwärmeleistung nach Baujahr-Klasse (Stack, nansum) + NaN-Hilfslinie

Korrekturen gegenüber Vorversion:
[1] list_smart_meter_files: schließt contextual_data.csv UND alle selected_*.csv aus,
    sodass Step-2-Outputs nicht fälschlich als Rohdaten eingelesen werden.
[2] Spitzenlast: kommt jetzt direkt aus dem tatsächlichen Maximum der stündlichen
    Effekt1-Mittelwerte (peak_kw_2019 = max über df_hourly je Meter). Maks.-effekt 1
    wird nicht mehr eingelesen oder verwendet.
[3] Pie Plot 3 (Anzahl Meter je Haustyp) und Pie Plot 5 (Anteil am Jahresenergiebedarf)
    haben eindeutige, unterschiedliche Titel, damit beide nicht verwechselt werden.
[4] Stack-Plots: NaN wird methodisch korrekt ignoriert (nansum). Eine zweite Y-Achse
    (rot, gestrichelt) zeigt den prozentualen Anteil fehlender Meter je Stunde.
    So sind Zeiträume mit schlechter Datenlage sofort erkennbar.
[5] construction_year NaN -> "Unbekannt". Im CHECK C wird der prozentuale Anteil
    fehlender Baujahre explizit ausgewiesen.

Hinweis zu Einheiten:
- Energi 1 Varmeenergi: kumulierter kWh-Zählerstand. Jahresverbrauch = max-min (2019).
- Effekt 1: Leistung in kW. Stündliche Mittelung bei Mehrfachmessungen.
  Peak = Maximum aller stündlichen Mittelwerte je Meter über 2019.
- NaN in Effekt 1: kein Messwert für diese Stunde. Wird in Step 1 NICHT interpoliert.
"""

from __future__ import annotations

import os
import glob
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# 0) KONFIGURATION – hier anpassen
# =============================================================================

DATA_DIR = r"src/ACES-2026/Data/Aalborg_smart_meter_data"
CONTEXT_FILENAME = "contextual_data.csv"

# Smart-Meter Spalten
COL_DATETIME = "RoundedReadTime"
COL_METER_ID = "MeterID"
COL_EFFEKT1  = "Effekt 1"              # kW – Leistung (Zeitreihe + Spitzenlast)
COL_ENERGI1  = "Energi 1 Varmeenergi"  # kWh – kumulierter Zählerstand (Jahresverbrauch)
# COL_MAX_EFFEKT entfernt: Spitzenlast wird direkt aus Effekt 1 berechnet

# Optional eingelesen, nicht geplottet
COL_T1 = "Temperatur 1"
COL_T2 = "Temperatur 2"

# Kontextdaten Spalten
CTX_METER_ID     = "meter_id"
CTX_UNIT_TYPE    = "unit_type"
CTX_CONSTRUCTION = "construction_year"

DAYFIRST = True

# Plot-Style
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

PLOTS_DIR       = "src/ACES-2026/plots"
LABEL_FONTSIZE  = 15
TICK_FONTSIZE   = 13
LEGEND_FONTSIZE = 13
TITLE_FONTSIZE  = 16

_PROJECT_PALETTE = [
    "#00395B",  # EUF-Blau
    "#C17A2F",  # Amber
    "#769D7B",  # Mint
    "#2F6B4F",  # Dunkelgrün
    "#C8A84B",  # Gold
    "#A0463A",  # Rot
    "#888888",  # Grau
]
_BAUJAHR_PALETTE = [
    "#2F6B4F", "#00395B", "#769D7B", "#C8A84B",
    "#C17A2F", "#A0463A", "#888888", "#1A1A1A",
]
_COLOR_NAN = "#A0463A"


def _ppt_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, color="#CCCCCC")
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.margins(x=0)


def _save_plot(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {path}")

# =============================================================================
# 1) HILFSFUNKTIONEN
@ -306,40 +337,30 @@ def compute_meter_stats(df_2019: pd.DataFrame, df_hourly: pd.DataFrame) -> pd.Da
# =============================================================================

def plot_stats_panel(meter_stats: pd.DataFrame) -> None:
    """
    3 Plots in einem Panel (Korrektur [2] + [3]):

    Plot 1 – Histogramm Jahresverbrauch (MWh) je unit_type
    Plot 2 – Scatter Jahresverbrauch vs. Spitzenlast je unit_type
             Jahresverbrauch: aus Energiezähler (unabhängige Quelle)
             Spitzenlast:     tatsächlicher Peak aus Effekt 1 (stündlich)
    Plot 3 – Pie: ANZAHL METER je unit_type
             -> zeigt die Haustyp-Struktur des Datensatzes
             -> NICHT den Energiebeitrag (das ist Pie in Plot 5)
    """
    unit_types_sorted = sorted(meter_stats["unit_type"].fillna("unbekannt").unique().tolist())
    palette    = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4", "#607D8B"]
    type_color = {t: palette[i % len(palette)] for i, t in enumerate(unit_types_sorted)}
    type_color = {t: _PROJECT_PALETTE[i % len(_PROJECT_PALETTE)] for i, t in enumerate(unit_types_sorted)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor="white")
    fig.suptitle(
        "Aalborg Smart Meter – Kennwerte 2019 je Haustyp\n"
        "(Jahresverbrauch: Energiezähler max−min  |  Spitzenlast: tatsächlicher Effekt-1-Peak)",
        fontsize=12, fontweight="bold"
        "Aalborg Smart Meter – Annual statistics 2019 by building type\n"
        "(Annual consumption: energy meter max−min  |  Peak load: actual Effekt-1 peak)",
        fontsize=TITLE_FONTSIZE, fontweight="bold", color="#1A1A1A"
    )

    # --- Plot 1: Histogramm Jahresverbrauch ---
    # --- Plot 1: Histogram annual consumption ---
    ax = axes[0]
    for ut in unit_types_sorted:
        sub = meter_stats.loc[meter_stats["unit_type"] == ut, "annual_kwh_2019"].dropna() / 1000
        if len(sub):
            ax.hist(sub, bins=30, color=type_color[ut], alpha=0.7, label=ut)
    ax.set_xlabel("Jahresverbrauch [MWh/Jahr]")
    ax.set_ylabel("Anzahl Meter")
    ax.set_title("Verteilung Jahresverbrauch\n(Quelle: Energiezähler max−min)")
    ax.legend(fontsize=8)

    # --- Plot 2: Scatter Jahresverbrauch vs. tatsächliche Spitzenlast ---
    ax.set_xlabel("Annual consumption in MWh/year", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Number of meters", fontsize=LABEL_FONTSIZE)
    ax.set_title("Distribution of annual consumption\n(Source: energy meter max−min)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax)
    ax.margins(x=0.04)

    # --- Plot 2: Scatter annual consumption vs. peak load ---
    ax = axes[1]
    for ut in unit_types_sorted:
        grp = meter_stats[meter_stats["unit_type"] == ut].dropna(
@ -351,14 +372,14 @@ def plot_stats_panel(meter_stats: pd.DataFrame) -> None:
                grp["peak_kw_2019"],
                color=type_color[ut], s=10, alpha=0.6, label=ut
            )
    ax.set_xlabel("Jahresverbrauch [MWh/Jahr]\n(Quelle: Energiezähler)")
    ax.set_ylabel("Spitzenlast [kW]\n(Quelle: max. stündl. Effekt 1)")
    ax.set_title("Jahresverbrauch vs. Spitzenlast\n(beide Quellen unabhängig voneinander)")
    ax.legend(fontsize=8)

    # --- Plot 3: Pie – ANZAHL METER je unit_type ---
    # Korrektur [3]: Dieser Pie zeigt Haustyp-Struktur (Meter-Anzahl).
    # Bewusst verschieden von Pie in Plot 5, der Energieanteile zeigt.
    ax.set_xlabel("Annual consumption in MWh/year\n(Source: energy meter)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Peak load in kW\n(Source: max. hourly Effekt 1)", fontsize=LABEL_FONTSIZE)
    ax.set_title("Annual consumption vs. peak load\n(independent sources)", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax)
    ax.margins(x=0.04)

    # --- Plot 3: Pie – number of meters per building type ---
    ax = axes[2]
    counts = (
        meter_stats["unit_type"].fillna("unbekannt")
@ -373,16 +394,15 @@ def plot_stats_panel(meter_stats: pd.DataFrame) -> None:
        colors=[type_color[t] for t in counts.index],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
        textprops={"fontsize": TICK_FONTSIZE},
    )
    ax.set_title(
        "Haustyp-Struktur: Anteil Meter je Typ\n"
        "(Anzahl Meter – NICHT Energiebeitrag)\n"
        "→ vgl. Energieanteil: Plot 5",
        fontsize=8, fontweight="bold"
        "Building type: share of meters\n(meter count – NOT energy share)",
        fontsize=TICK_FONTSIZE, fontweight="bold", color="#1A1A1A"
    )

    plt.tight_layout()
    fig.tight_layout()
    _save_plot(fig, "step1_stats_panel.png")
    plt.show()


@ -392,31 +412,15 @@ def _nan_fraction_per_hour(df_hourly: pd.DataFrame) -> pd.Series:


def plot_series_panels(df_hourly: pd.DataFrame, df_meta: pd.DataFrame) -> None:
    """
    Plots 4–6: Zeitreihen der Wärmeleistung 2019.

    NaN-Behandlung (Korrektur [4]):
    - Stündliche Summen werden mit nansum berechnet: Fehlende Meter tragen 0 bei,
      vollständig fehlende Stunden bleiben NaN.
    - Dies kann die tatsächliche Last in Stunden mit vielen fehlenden Metern
      unterschätzen. Deshalb zeigt jeder Stack-Plot eine zweite Y-Achse (rot,
      gestrichelt) mit dem prozentualen Anteil fehlender Meter je Stunde.
    - Hohe rote Werte = Ergebnis in diesem Zeitraum mit Vorsicht interpretieren.
    """
    # Gesamtlast (min_count=1: vollständig fehlende Stunden bleiben NaN)
    total_heat = df_hourly.sum(axis=1, min_count=1)
    nan_frac   = _nan_fraction_per_hour(df_hourly)

    # NaN-Anteil je Stunde (Prozent)
    nan_frac = _nan_fraction_per_hour(df_hourly)

    # Aggregation nach unit_type
    heat_by_type: dict[str, pd.Series] = {}
    for ut, mids in df_meta.groupby("unit_type")["meter_id"]:
        cols = [m for m in mids if m in df_hourly.columns]
        if cols:
            heat_by_type[ut] = df_hourly[cols].sum(axis=1, min_count=1)

    # Aggregation nach Baujahr-Klasse
    BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                     "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]
    heat_by_baujahr: dict[str, pd.Series] = {}
@ -425,59 +429,51 @@ def plot_series_panels(df_hourly: pd.DataFrame, df_meta: pd.DataFrame) -> None:
        if cols:
            heat_by_baujahr[bk] = df_hourly[cols].sum(axis=1, min_count=1)

    # Farbpaletten
    types_sorted = sorted(heat_by_type.keys())
    palette      = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4", "#607D8B"]
    color_map    = {t: palette[i % len(palette)] for i, t in enumerate(types_sorted)}
    b_palette    = ["#7B1FA2", "#C62828", "#EF6C00", "#F9A825", "#558B2F", "#0277BD", "#00838F", "#9E9E9E"]
    b_color      = {bk: b_palette[i % len(b_palette)] for i, bk in enumerate(BAUJAHR_ORDER)}
    color_map = {t: _PROJECT_PALETTE[i % len(_PROJECT_PALETTE)] for i, t in enumerate(types_sorted)}
    b_color   = {bk: _BAUJAHR_PALETTE[i % len(_BAUJAHR_PALETTE)] for i, bk in enumerate(BAUJAHR_ORDER)}

    def _add_nan_axis(ax_main: plt.Axes) -> plt.Axes:
        """Fügt zweite Y-Achse mit NaN-Anteil (rot, gestrichelt) hinzu."""
    def _add_nan_axis(ax_main):
        ax_r = ax_main.twinx()
        ax_r.fill_between(df_hourly.index, nan_frac.values, alpha=0.12, color="red")
        ax_r.plot(df_hourly.index, nan_frac.values, color="red", linewidth=0.6,
                  linestyle="--", label="Fehlende Meter [%]")
        ax_r.set_ylabel("Anteil fehlender Meter [%]", color="red", fontsize=8)
        ax_r.tick_params(axis="y", labelcolor="red")
        ax_r.fill_between(df_hourly.index, nan_frac.values, alpha=0.12, color=_COLOR_NAN)
        ax_r.plot(df_hourly.index, nan_frac.values, color=_COLOR_NAN, linewidth=0.6, linestyle="--")
        ax_r.set_ylabel("Missing meters in %", color=_COLOR_NAN, fontsize=TICK_FONTSIZE)
        ax_r.tick_params(axis="y", labelcolor=_COLOR_NAN)
        ax_r.set_ylim(0, 100)
        return ax_r

    # -------------------------------------------------------------------------
    # Plot 4: Gesamtlast + NaN-Anteil
    # Plot 4: Total heat load + NaN share
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df_hourly.index, total_heat.values, color="#1f77b4", linewidth=0.7)
    ax1.set_title(
        "Gesamtwärmeleistung (Summe Effekt 1, NaN ignoriert) – 2019\n"
        "Rote Linie: Anteil Meter ohne Messwert je Stunde",
        fontweight="bold"
    )
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Wärmeleistung [kW]")
    fig1, ax1 = plt.subplots(figsize=(16, 9), facecolor="white")
    ax1.plot(df_hourly.index, total_heat.values, color="#00395B", linewidth=0.7)
    ax1.set_title("Total heat load (sum Effekt 1, NaN ignored) – 2019",
                  fontsize=TITLE_FONTSIZE, fontweight="bold", color="#1A1A1A")
    ax1.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("Heat load in kW", fontsize=LABEL_FONTSIZE)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _add_nan_axis(ax1)
    _ppt_style(ax1)

    ax1.annotate(
        f"Gesamt-NaN-Anteil (Stunden ohne jeden Wert): {total_heat.isna().mean():.2%}\n"
        f"Ø NaN-Meter je Stunde: {nan_frac.mean():.1f}%\n"
        f"Max Gesamtlast: {np.nanmax(total_heat.values):.1f} kW  "
        f"| Ø Gesamtlast: {np.nanmean(total_heat.values):.1f} kW",
        f"Total NaN share (hours without any value): {total_heat.isna().mean():.2%}\n"
        f"Avg. missing meters per hour: {nan_frac.mean():.1f}%\n"
        f"Max total load: {np.nanmax(total_heat.values):.1f} kW  "
        f"| Avg total load: {np.nanmean(total_heat.values):.1f} kW",
        xy=(0.01, 0.97), xycoords="axes fraction", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        fontsize=8,
        fontsize=TICK_FONTSIZE,
    )
    fig1.tight_layout()
    _save_plot(fig1, "step1_total_heat_load.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Plot 5: Stack nach unit_type + Pie nach Jahresenergie
    # Korrektur [3]: Pie hier zeigt ENERGIEBEITRAG je Haustyp (nicht Meter-Anzahl).
    # Titel macht den Unterschied zu Plot 3 (Meter-Anzahl) explizit sichtbar.
    # Plot 5: Stack by building type + pie by annual energy
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 1]})

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 9), facecolor="white",
                                       gridspec_kw={"width_ratios": [3, 1]})
    bottom = np.zeros(len(df_hourly.index))
    for t in types_sorted:
        vals = np.nan_to_num(heat_by_type[t].to_numpy(), nan=0.0)
@ -486,18 +482,15 @@ def plot_series_panels(df_hourly: pd.DataFrame, df_meta: pd.DataFrame) -> None:
        bottom += vals

    _add_nan_axis(ax2a)
    ax2a.set_title(
        "Gesamtwärmeleistung nach Haustyp – 2019\n"
        "(Rote Linie: Anteil Meter ohne Messwert je Stunde)",
        fontweight="bold"
    )
    ax2a.set_xlabel("Datum")
    ax2a.set_ylabel("Wärmeleistung [kW]")
    ax2a.set_title("Heat load by building type – 2019",
                   fontsize=TITLE_FONTSIZE, fontweight="bold", color="#1A1A1A")
    ax2a.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax2a.set_ylabel("Heat load in kW", fontsize=LABEL_FONTSIZE)
    ax2a.xaxis.set_major_locator(mdates.MonthLocator())
    ax2a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2a.legend(fontsize=9, loc="upper right")
    ax2a.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=False)
    _ppt_style(ax2a)

    # Pie: ENERGIEBEITRAG je Haustyp (bewusst verschieden von Plot 3: Meter-Anzahl)
    pie_labels, pie_vals, pie_cols = [], [], []
    for t in types_sorted:
        e = float(np.nansum(heat_by_type[t].values))
@ -505,22 +498,19 @@ def plot_series_panels(df_hourly: pd.DataFrame, df_meta: pd.DataFrame) -> None:
            pie_labels.append(t)
            pie_vals.append(e)
            pie_cols.append(color_map[t])

    ax2b.pie(pie_vals, labels=pie_labels, colors=pie_cols,
             autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax2b.set_title(
        "Anteil am Jahreswärmebedarf\nje Haustyp (Effekt-1-Energie)\n"
        "≠ Plot 3 (Anteil Meter-Anzahl)",
        fontsize=8, fontweight="bold"
    )
             autopct="%1.1f%%", startangle=90, textprops={"fontsize": TICK_FONTSIZE})
    ax2b.set_title("Share of annual heat demand\nby building type (Effekt-1 energy)",
                   fontsize=TICK_FONTSIZE, fontweight="bold", color="#1A1A1A")

    fig2.tight_layout()
    _save_plot(fig2, "step1_heat_by_type.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Plot 6: Stack nach Baujahr-Klasse + NaN-Anteil
    # Plot 6: Stack by construction year class + NaN share
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(16, 6))
    fig3, ax3 = plt.subplots(figsize=(16, 9), facecolor="white")
    bottom = np.zeros(len(df_hourly.index))
    for bk in BAUJAHR_ORDER:
        if bk not in heat_by_baujahr:
@ -531,38 +521,38 @@
        bottom += vals

    _add_nan_axis(ax3)
    ax3.set_title(
        "Gesamtwärmeleistung nach Baujahr-Klasse – 2019\n"
        "(Rote Linie: Anteil Meter ohne Messwert je Stunde)",
        fontweight="bold"
    )
    ax3.set_xlabel("Datum")
    ax3.set_ylabel("Wärmeleistung [kW]")
    ax3.set_title("Heat load by construction year class – 2019",
                  fontsize=TITLE_FONTSIZE, fontweight="bold", color="#1A1A1A")
    ax3.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel("Heat load in kW", fontsize=LABEL_FONTSIZE)
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax3.legend(fontsize=9, loc="upper right", title="Baujahr")
    ax3.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=False,
               title="Construction year", title_fontsize=LEGEND_FONTSIZE)
    _ppt_style(ax3)

    fig3.tight_layout()
    _save_plot(fig3, "step1_heat_by_year.png")
    plt.show()


# =============================================================================
# 3) MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("ACES – Step 1: Aalborg 2019 (methodisch korrigiert)")
    print("=" * 65)

    # Dateiliste: nur Rohdaten (Korrektur [1])
    files = list_smart_meter_files(DATA_DIR)
    print(f"Gefundene Smart-Meter-Rohdaten-Dateien: {len(files)}")
    for f in files:
        print(f"  {os.path.basename(f)}")

    # CHECK A: Zeitabdeckung je Datei
    print("\nCHECK A: Datei-Zeitabdeckung (min/max pro Datei)")
    cov = quick_file_time_coverage(files)
    print(cov.to_string(index=False))

