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
# =============================================================================

def detect_csv_separator(path: str, default: str = ",") -> str:
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        return default


def list_smart_meter_files(data_dir: str) -> list[str]:
    """
    Gibt ausschließlich Rohdaten-CSVs der Smart-Meter zurück.

    Korrektur [1]:
    Ausgeschlossen werden:
    - contextual_data.csv  (Kontextdatei, kein Smart-Meter-Format)
    - selected_*.csv       (Step-2-Outputs – würden mit Spaltenfehler abstürzen)

    So läuft Step 1 korrekt, unabhängig davon, ob Step 2 bereits gelaufen ist.
    """
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    files = [
        f for f in files
        if os.path.basename(f) != CONTEXT_FILENAME
        and not os.path.basename(f).startswith("selected_")
    ]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"Keine Smart-Meter-CSV-Dateien in '{data_dir}' gefunden.\n"
            "(contextual_data.csv und selected_*.csv werden ignoriert.)"
        )
    return files


def quick_file_time_coverage(files: list[str]) -> pd.DataFrame:
    rows = []
    for f in files:
        sep = detect_csv_separator(f)
        d = pd.read_csv(f, sep=sep, usecols=[COL_DATETIME, COL_METER_ID], low_memory=False)
        t = pd.to_datetime(d[COL_DATETIME], errors="coerce", dayfirst=DAYFIRST)
        rows.append({
            "file": os.path.basename(f),
            "sep": sep,
            "rows": len(d),
            "meters": d[COL_METER_ID].astype(str).nunique(),
            "min_time": t.min(),
            "max_time": t.max(),
            "nat_share": float(t.isna().mean()),
        })
    return pd.DataFrame(rows).sort_values(["min_time", "file"], na_position="last")


def load_context() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, CONTEXT_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kontextdatei nicht gefunden: {path}")
    sep = detect_csv_separator(path)
    print(f"Lese Kontextdaten: {CONTEXT_FILENAME} (sep='{sep}')")
    df_ctx = pd.read_csv(path, sep=sep, low_memory=False)
    print(f"  Kontext-Spalten: {list(df_ctx.columns)}")

    missing = [c for c in [CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION] if c not in df_ctx.columns]
    if missing:
        raise KeyError(
            "Kontext-Spalten fehlen: " + ", ".join(missing) +
            "\nBitte CTX_* Variablen oben an die echten Spaltennamen anpassen."
        )

    df_ctx = df_ctx[[CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION]].copy()
    df_ctx.columns = ["meter_id", "unit_type", "construction_year"]
    df_ctx["meter_id"] = df_ctx["meter_id"].astype(str).str.strip()
    df_ctx["unit_type"] = df_ctx["unit_type"].astype(str)
    # Fehlende Baujahre bleiben NaN; baujahr_klasse() klassifiziert sie als "Unbekannt"
    df_ctx["construction_year"] = pd.to_numeric(df_ctx["construction_year"], errors="coerce")
    return df_ctx


def baujahr_klasse(year: float) -> str:
    """Klassifiziert Baujahr in Periode. NaN -> 'Unbekannt'. (Korrektur [5])"""
    if pd.isna(year):
        return "Unbekannt"
    y = int(year)
    if y < 1919: return "vor 1919"
    if y < 1949: return "1919–1948"
    if y < 1969: return "1949–1968"
    if y < 1979: return "1969–1978"
    if y < 1995: return "1979–1994"
    if y < 2010: return "1995–2009"
    return "ab 2010"


def parse_numeric_series(s: pd.Series) -> pd.Series:
    """Robust numerisch: Dezimalkomma, Einheiten/Strings -> float."""
    s2 = s.astype(str).str.replace(",", ".", regex=False)
    s2 = s2.str.replace(r"[^\d\.\-]", "", regex=True)
    s2 = s2.replace("", np.nan)
    return pd.to_numeric(s2, errors="coerce")


def load_and_filter_2019(files: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Einlese-Pass: lädt benötigte Spalten, filtert auf 2019.

    Korrektur [2]: Maks.-effekt 1 wird nicht mehr eingelesen.
    Spalten im Output df_2019: datetime, meter_id, effekt1_kw, energi1_kwh, t1_c, t2_c
    """
    need = [COL_DATETIME, COL_METER_ID, COL_EFFEKT1, COL_ENERGI1, COL_T1, COL_T2]

    chunks_2019 = []
    info_rows   = []

    for i, f in enumerate(files, 1):
        sep = detect_csv_separator(f)
        print(f"  Lese {i:02d}/{len(files)}: {os.path.basename(f)} (sep='{sep}')")

        d = pd.read_csv(f, sep=sep, usecols=need, low_memory=False)
        d = d.rename(columns={
            COL_DATETIME: "datetime",
            COL_METER_ID: "meter_id",
            COL_EFFEKT1:  "effekt1_kw",
            COL_ENERGI1:  "energi1_kwh",
            COL_T1: "t1_c",
            COL_T2: "t2_c",
        })

        d["meter_id"]    = d["meter_id"].astype(str).str.strip()
        d["datetime"]    = pd.to_datetime(d["datetime"], errors="coerce", dayfirst=DAYFIRST)
        d["effekt1_kw"]  = parse_numeric_series(d["effekt1_kw"])
        d["energi1_kwh"] = parse_numeric_series(d["energi1_kwh"])
        d["t1_c"]        = parse_numeric_series(d["t1_c"])
        d["t2_c"]        = parse_numeric_series(d["t2_c"])

        nat_share = float(d["datetime"].isna().mean())
        d = d.dropna(subset=["datetime", "meter_id"])
        d_2019 = d[d["datetime"].dt.year == 2019].copy()

        info_rows.append({
            "file": os.path.basename(f),
            "sep": sep,
            "rows": len(d),
            "rows_2019": len(d_2019),
            "meters_2019": int(d_2019["meter_id"].nunique()),
            "datetime_nat_share": nat_share,
        })
        chunks_2019.append(d_2019)

    df_2019 = pd.concat(chunks_2019, ignore_index=True)
    df_info = pd.DataFrame(info_rows)
    return df_2019, df_info


def check_2019_coverage(df_2019: pd.DataFrame) -> None:
    df = df_2019[["meter_id", "datetime"]].copy()
    df["datetime_h"] = df["datetime"].dt.floor("h")

    expected_hours   = 8760
    hours_per_meter  = df.groupby("meter_id")["datetime_h"].nunique()
    missing_hours    = expected_hours - hours_per_meter

    print("\nCHECK B1: Stunden-Abdeckung 2019 je Meter")
    print(f"  Meter insgesamt (mit Daten in 2019): {hours_per_meter.size}")
    print(f"  Meter mit vollständigen 8760h:       {(missing_hours == 0).sum()}")
    print(f"  Meter mit fehlenden Stunden:         {(missing_hours > 0).sum()}")
    print(f"  Max fehlende Stunden (worst case):   {missing_hours.max()}")
    print("  Top 10 fehlende Stunden:")
    print(missing_hours.sort_values(ascending=False).head(10))

    counts_per_hour = df.groupby(["meter_id", "datetime_h"]).size()
    print("\nCHECK B2: Mehrfachmessungen pro Stunde?")
    print(counts_per_hour.describe())
    print("  Anteil Stunden mit >1 Messung:", float((counts_per_hour > 1).mean()))


def build_hourly_matrix(df_2019: pd.DataFrame) -> pd.DataFrame:
    """
    Stündliche Effekt1-Matrix: 8760 Zeilen x n_meter Spalten.
    Mehrfachmessungen je Stunde werden gemittelt. NaN bleibt NaN (keine Interpolation).
    """
    df = df_2019[["datetime", "meter_id", "effekt1_kw"]].copy()
    df["datetime_h"] = df["datetime"].dt.floor("h")

    df_agg = df.groupby(["datetime_h", "meter_id"], as_index=False)["effekt1_kw"].mean()

    df_hourly = df_agg.pivot(index="datetime_h", columns="meter_id", values="effekt1_kw")
    full_index = pd.date_range("2019-01-01 00:00", "2019-12-31 23:00", freq="h")
    df_hourly  = df_hourly.reindex(full_index)
    df_hourly.index.name = "datetime"
    return df_hourly


def compute_meter_stats(df_2019: pd.DataFrame, df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Jahresverbrauch und tatsächliche Spitzenlast pro Meter für 2019.

    Jahresverbrauch (annual_kwh_2019):
      Aus Energi 1 Varmeenergi (kumulierter Zählerstand): max - min innerhalb 2019.
      Diese Berechnung ist völlig unabhängig von Effekt 1 und von NaN-Lücken dort
      unberührt. (Keine Änderung gegenüber Vorversion.)

    Spitzenlast (peak_kw_2019) – Korrektur [2]:
      Tatsächlicher Maximalwert der stündlichen Effekt1-Mittelwerte (df_hourly).
      Das ist der höchste real gemessene Leistungswert je Meter in 2019.
      NaN-Stunden werden ignoriert (skipna=True). Kein Rückgriff mehr auf Maks.-effekt 1.
    """
    # Jahresverbrauch aus kumuliertem Energiezähler (unabhängig von Effekt1)
    e = df_2019.dropna(subset=["energi1_kwh"]).groupby("meter_id")["energi1_kwh"].agg(["min", "max"])
    annual_kwh = (e["max"] - e["min"]).rename("annual_kwh_2019")

    # Tatsächliche Spitzenlast: max der stündlichen Effekt1-Mittelwerte
    peak_kw = df_hourly.max(axis=0, skipna=True).rename("peak_kw_2019")

    out = pd.concat([annual_kwh, peak_kw], axis=1)
    out["annual_kwh_negative"] = out["annual_kwh_2019"] < 0
    return out


# =============================================================================
# 2) PLOT-FUNKTIONEN
# =============================================================================

def plot_stats_panel(meter_stats: pd.DataFrame) -> None:
    unit_types_sorted = sorted(meter_stats["unit_type"].fillna("unbekannt").unique().tolist())
    type_color = {t: _PROJECT_PALETTE[i % len(_PROJECT_PALETTE)] for i, t in enumerate(unit_types_sorted)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor="white")
    fig.suptitle(
        "Aalborg Smart Meter – Annual statistics 2019 by building type\n"
        "(Annual consumption: energy meter max−min  |  Peak load: actual Effekt-1 peak)",
        fontsize=TITLE_FONTSIZE, fontweight="bold", color="#1A1A1A"
    )

    # --- Plot 1: Histogram annual consumption ---
    ax = axes[0]
    for ut in unit_types_sorted:
        sub = meter_stats.loc[meter_stats["unit_type"] == ut, "annual_kwh_2019"].dropna() / 1000
        if len(sub):
            ax.hist(sub, bins=30, color=type_color[ut], alpha=0.7, label=ut)
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
            subset=["annual_kwh_2019", "peak_kw_2019"]
        )
        if len(grp):
            ax.scatter(
                grp["annual_kwh_2019"] / 1000,
                grp["peak_kw_2019"],
                color=type_color[ut], s=10, alpha=0.6, label=ut
            )
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
        .value_counts()
        .reindex(unit_types_sorted)
        .fillna(0)
        .astype(int)
    )
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=[type_color[t] for t in counts.index],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": TICK_FONTSIZE},
    )
    ax.set_title(
        "Building type: share of meters\n(meter count – NOT energy share)",
        fontsize=TICK_FONTSIZE, fontweight="bold", color="#1A1A1A"
    )

    fig.tight_layout()
    _save_plot(fig, "step1_stats_panel.png")
    plt.show()


def _nan_fraction_per_hour(df_hourly: pd.DataFrame) -> pd.Series:
    """Anteil NaN-Meter je Stunde in Prozent (0% = alle vorhanden, 100% = alle NaN)."""
    return df_hourly.isna().mean(axis=1) * 100


def plot_series_panels(df_hourly: pd.DataFrame, df_meta: pd.DataFrame) -> None:
    total_heat = df_hourly.sum(axis=1, min_count=1)
    nan_frac   = _nan_fraction_per_hour(df_hourly)

    heat_by_type: dict[str, pd.Series] = {}
    for ut, mids in df_meta.groupby("unit_type")["meter_id"]:
        cols = [m for m in mids if m in df_hourly.columns]
        if cols:
            heat_by_type[ut] = df_hourly[cols].sum(axis=1, min_count=1)

    BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                     "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]
    heat_by_baujahr: dict[str, pd.Series] = {}
    for bk, mids in df_meta.groupby("baujahr_klasse")["meter_id"]:
        cols = [m for m in mids if m in df_hourly.columns]
        if cols:
            heat_by_baujahr[bk] = df_hourly[cols].sum(axis=1, min_count=1)

    types_sorted = sorted(heat_by_type.keys())
    color_map = {t: _PROJECT_PALETTE[i % len(_PROJECT_PALETTE)] for i, t in enumerate(types_sorted)}
    b_color   = {bk: _BAUJAHR_PALETTE[i % len(_BAUJAHR_PALETTE)] for i, bk in enumerate(BAUJAHR_ORDER)}

    def _add_nan_axis(ax_main):
        ax_r = ax_main.twinx()
        ax_r.fill_between(df_hourly.index, nan_frac.values, alpha=0.12, color=_COLOR_NAN)
        ax_r.plot(df_hourly.index, nan_frac.values, color=_COLOR_NAN, linewidth=0.6, linestyle="--")
        ax_r.set_ylabel("Missing meters in %", color=_COLOR_NAN, fontsize=TICK_FONTSIZE)
        ax_r.tick_params(axis="y", labelcolor=_COLOR_NAN)
        ax_r.set_ylim(0, 100)
        return ax_r

    # -------------------------------------------------------------------------
    # Plot 4: Total heat load + NaN share
    # -------------------------------------------------------------------------
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
        f"Total NaN share (hours without any value): {total_heat.isna().mean():.2%}\n"
        f"Avg. missing meters per hour: {nan_frac.mean():.1f}%\n"
        f"Max total load: {np.nanmax(total_heat.values):.1f} kW  "
        f"| Avg total load: {np.nanmean(total_heat.values):.1f} kW",
        xy=(0.01, 0.97), xycoords="axes fraction", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        fontsize=TICK_FONTSIZE,
    )
    fig1.tight_layout()
    _save_plot(fig1, "step1_total_heat_load.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Plot 5: Stack by building type + pie by annual energy
    # -------------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 9), facecolor="white",
                                       gridspec_kw={"width_ratios": [3, 1]})
    bottom = np.zeros(len(df_hourly.index))
    for t in types_sorted:
        vals = np.nan_to_num(heat_by_type[t].to_numpy(), nan=0.0)
        ax2a.fill_between(df_hourly.index, bottom, bottom + vals,
                          alpha=0.75, color=color_map[t], label=t)
        bottom += vals

    _add_nan_axis(ax2a)
    ax2a.set_title("Heat load by building type – 2019",
                   fontsize=TITLE_FONTSIZE, fontweight="bold", color="#1A1A1A")
    ax2a.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax2a.set_ylabel("Heat load in kW", fontsize=LABEL_FONTSIZE)
    ax2a.xaxis.set_major_locator(mdates.MonthLocator())
    ax2a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2a.legend(fontsize=LEGEND_FONTSIZE, loc="upper right", frameon=False)
    _ppt_style(ax2a)

    pie_labels, pie_vals, pie_cols = [], [], []
    for t in types_sorted:
        e = float(np.nansum(heat_by_type[t].values))
        if e > 0:
            pie_labels.append(t)
            pie_vals.append(e)
            pie_cols.append(color_map[t])
    ax2b.pie(pie_vals, labels=pie_labels, colors=pie_cols,
             autopct="%1.1f%%", startangle=90, textprops={"fontsize": TICK_FONTSIZE})
    ax2b.set_title("Share of annual heat demand\nby building type (Effekt-1 energy)",
                   fontsize=TICK_FONTSIZE, fontweight="bold", color="#1A1A1A")

    fig2.tight_layout()
    _save_plot(fig2, "step1_heat_by_type.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Plot 6: Stack by construction year class + NaN share
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(16, 9), facecolor="white")
    bottom = np.zeros(len(df_hourly.index))
    for bk in BAUJAHR_ORDER:
        if bk not in heat_by_baujahr:
            continue
        vals = np.nan_to_num(heat_by_baujahr[bk].to_numpy(), nan=0.0)
        ax3.fill_between(df_hourly.index, bottom, bottom + vals,
                         alpha=0.75, color=b_color[bk], label=bk)
        bottom += vals

    _add_nan_axis(ax3)
    ax3.set_title("Heat load by construction year class – 2019",
                  fontsize=TITLE_FONTSIZE, fontweight="bold", color="#1A1A1A")
    ax3.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel("Heat load in kW", fontsize=LABEL_FONTSIZE)
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
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

    # Kontext laden
    df_ctx = load_context()

    # Einlesen + Filter 2019
    print("\nSchritt 1: Einlesen + Filter 2019 ...")
    df_2019, df_info = load_and_filter_2019(files)
    print(f"  Zeilen 2019 (gesamt): {len(df_2019):,}")
    print(f"  Unique meter_id 2019: {df_2019['meter_id'].nunique():,}")

    if len(df_2019) == 0:
        raise ValueError("df_2019 ist leer – bitte COL_DATETIME / DAYFIRST prüfen.")

    # Kontext-Join
    df_meta = (
        df_2019[["meter_id"]].drop_duplicates()
        .merge(df_ctx, on="meter_id", how="left")
    )
    df_meta["unit_type"]      = df_meta["unit_type"].fillna("unbekannt")
    df_meta["baujahr_klasse"] = df_meta["construction_year"].apply(baujahr_klasse)

    # CHECK C: Kontext-Join-Qualität + Baujahr-Vollständigkeit (Korrektur [5])
    print("\nCHECK C: Kontext-Join-Qualität")
    n_total        = df_meta["meter_id"].nunique()
    n_missing_type = int((df_meta["unit_type"] == "unbekannt").sum())
    n_missing_year = int(df_meta["construction_year"].isna().sum())
    n_unknown_bj   = int((df_meta["baujahr_klasse"] == "Unbekannt").sum())
    print(f"  Meter in 2019:               {n_total}")
    print(f"  Fehlender unit_type:         {n_missing_type} ({n_missing_type/n_total:.1%})")
    print(f"  Fehlendes construction_year: {n_missing_year} ({n_missing_year/n_total:.1%})"
          " → werden als 'Unbekannt' klassifiziert")
    print(f"  Baujahr 'Unbekannt' gesamt:  {n_unknown_bj} ({n_unknown_bj/n_total:.1%})"
          " (beeinflusst Aussagekraft von Plot 6)")
    print("  unit_type-Verteilung:")
    print(df_meta["unit_type"].value_counts())

    # CHECK B: Stunden-Abdeckung
    check_2019_coverage(df_2019)

    # Stündliche Matrix (8760h x n_meter)
    print("\nSchritt 2: Stündliche Effekt1-Matrix (8760h) ...")
    df_hourly = build_hourly_matrix(df_2019)
    print(f"  Matrix: {df_hourly.shape[0]} Stunden x {df_hourly.shape[1]} Meter")

    # Meter-Stats: Jahresverbrauch (Energi1) + tatsächliche Spitzenlast (Effekt1-Peak)
    print("\nSchritt 3: Jahresverbrauch (Energiezähler) + tatsächliche Spitzenlast (Effekt-1-Peak) ...")
    stats = compute_meter_stats(df_2019, df_hourly)
    meter_stats = (
        df_meta.set_index("meter_id")
        .join(stats, how="left")
        .reset_index()
        .rename(columns={"index": "meter_id"})
    )

    print(f"  annual_kwh_2019 NaN:  {int(meter_stats['annual_kwh_2019'].isna().sum())}")
    print(f"  peak_kw_2019 NaN:     {int(meter_stats['peak_kw_2019'].isna().sum())}")
    n_neg = int((meter_stats["annual_kwh_2019"] < 0).sum())
    if n_neg:
        print(f"  Warnung: {n_neg} Meter mit negativer Jahresenergie (Zählerreset möglich)")

    # Plots 1–3: Statistik-Panel
    plot_stats_panel(meter_stats)

    # Plots 4–6: Zeitreihen mit NaN-Transparenz
    plot_series_panels(df_hourly, df_meta)


if __name__ == "__main__":
    main()
