#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 – Auswahl von 267 repräsentativen Lastprofilen (stratifiziert)

Ziel
----
Aus ~3000 Smart-Meter-Profilen werden 267 Profile ausgewählt, die die Aalborg-Realität
möglichst gut abbilden – passend zu einem Beispieldorf mit 267 Haushalten.

Methodik
--------
1) Meter mit unit_type "unclear" oder fehlend werden vor der Auswahl entfernt
   (Warnung zeigt wie viele und welche Meter wegfallen).
2) Meter mit zu geringer stündlicher Datenabdeckung in Effekt 1 werden entfernt
   (Mindestabdeckung konfigurierbar, Standard: 70% der 8760 Stunden = 6132h).
3) Exakt proportionale Stichprobe nach unit_type (Hamilton-Methode).
4) Innerhalb jedes unit_type: Stratifizierung nach Jahresverbrauch (Quantil-Bins).
5) Zufällige Auswahl innerhalb jeder Schicht (reproduzierbar über Seed).

Korrekturen gegenüber Vorversion
---------------------------------
[1]  Maks.-effekt 1 entfernt: Spitzenlast kommt direkt aus Effekt 1 (konsistent mit Step 1).
[2]  Mindestabdeckungsfilter: Meter mit < MIN_COVERAGE_FRAC Effekt1-Daten werden
     vor der Auswahl ausgeschlossen.
[3]  Warnung wenn Output < 267 Meter (fehlende Effekt1-Spalten).
[4]  Bin-Berechnung nur einmal, Ergebnis wird ans Meta-CSV weitergegeben.
[5]  Typ-Konsistenz im Fallback-Block: alle meter_id als str.
[6]  Statistische Validierung: Quantilvergleich Sample vs. Population (gedruckt).
[7]  Spaltenprüfung beim Einlesen: Dateien ohne erwartete Spalten werden
     übersprungen und gemeldet, statt mit Fehler abzubrechen.
[8]  unit_type "unclear" wird entfernt + Warnung.
[9]  Output enthält Jahresverbrauch (annual_kwh_2019) und unit_type je Meter.
[10] Output sortiert nach Jahresverbrauch absteigend.
[11] Alle 8 Step-1-Plots (inkl. 7-Tage gleitendem Mittelwert und Tagesprofilen je
     Jahreszeit) werden für die 267 ausgewählten Meter reproduziert.

Outputs
-------
- CSV long:  [datetime, meter_id, unit_type, annual_kwh_2019, effekt1_kw]  (267 Meter, 2019)
- CSV meta:  meter_id, unit_type, construction_year, annual_kwh_2019, peak_kw_2019,
             chosen, bin  –  sortiert nach annual_kwh_2019 absteigend
"""

from __future__ import annotations

import os
import glob
import csv

import numpy as np
import pandas as pd

# Plots in einem eigenen Fenster öffnen UND automatisch als PNG speichern (siehe
# PLOTS_DIR/_save unten). Ein interaktives GUI-Backend wird vor dem pyplot-Import
# gesetzt; das erste verfügbare Backend wird verwendet.
import matplotlib
# Try GUI backends; fall back to file-only Agg if none load successfully.
# Note: matplotlib.use() may not raise immediately – the import happens lazily
# at first figure creation, so we test each backend with a dummy figure.
for _backend in ("MacOSX", "TkAgg", "Qt5Agg", "QtAgg"):
    try:
        matplotlib.use(_backend, force=True)
        import matplotlib.pyplot as _plt_probe
        _fig = _plt_probe.figure()
        _plt_probe.close(_fig)
        break
    except Exception:
        continue
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats

# =============================================================================
# 0) KONFIGURATION
# =============================================================================

# Pfad relativ zum Skript-Standort auflösen, damit das Working-Directory egal ist
# (z.B. Spyder %runfile --wdir setzt das CWD auf den Skriptordner, nicht die Repo-Wurzel).
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # __file__ ggf. nicht definiert (interaktiv)
    _SCRIPT_DIR = os.getcwd()
_ACES_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # .../src/ACES-2026
DATA_DIR = os.path.join(_ACES_DIR, "Data", "Aalborg_smart_meter_data")
CONTEXT_FILENAME = "contextual_data.csv"

# Smart-Meter Spalten
COL_DATETIME = "RoundedReadTime"
COL_METER_ID = "MeterID"
COL_EFFEKT1  = "Effekt 1"              # kW – Leistung (Zeitreihe + Spitzenlast)
COL_ENERGI1  = "Energi 1 Varmeenergi"  # kWh – kumulierter Zählerstand (Jahresverbrauch)
# COL_MAX_EFFEKT entfernt [1]: Spitzenlast direkt aus Effekt 1

# Kontextdaten
CTX_METER_ID     = "meter_id"
CTX_UNIT_TYPE    = "unit_type"
CTX_CONSTRUCTION = "construction_year"

DAYFIRST = True

# Sampling
N_TARGET   = 267
N_BINS     = 10
RANDOM_SEED = 42                      # Fixed for reproducability

# Mindestabdeckung Effekt 1 [2]
# Meter mit weniger als MIN_COVERAGE_FRAC * 8760 Stunden Effekt1-Daten werden
# vor der Auswahl ausgeschlossen (Standard: 70% = 6132 Stunden).
MIN_COVERAGE_FRAC  = 0.70
MIN_COVERAGE_HOURS = int(MIN_COVERAGE_FRAC * 8760)  # = 6132

# unit_type-Werte, die vor der Auswahl ausgeschlossen werden [8]
EXCLUDE_UNIT_TYPES = {"unclear"}

# NaN/Interpolation Policy
INTERP_MAX_GAP_HOURS = 6

# Output
OUT_LONG = os.path.join(DATA_DIR, "selected_267_profiles_2019_long.csv")
OUT_META = os.path.join(DATA_DIR, "selected_267_profiles_meta.csv")
# Wide-Format: Datum + Smart-Meter 1..267 (1 = höchster Jahresverbrauch),
# stündliche kW-Werte. Keine echte meter_id / unit_type / annual_kwh.
OUT_WIDE = os.path.join(DATA_DIR, "selected_267_profiles_2019_wide.csv")

# Ordner zum automatischen Speichern der Plots (gleiche Konvention wie funcs/plots.py)
PLOTS_DIR = os.path.join(_ACES_DIR, "plots")


def _save(fig, filename):
    """Speichert die Figure als PNG im gemeinsamen Plots-Ordner."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {path}")

# Plot-Style
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    # Schrifttyp Calibri (Fallback auf DejaVu Sans, falls Calibri nicht installiert)
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Carlito", "DejaVu Sans"],
})

# Anzeigenamen für unit_type (Unterstriche raus, englische Bezeichnungen)
UNIT_TYPE_LABELS = {
    "apartment":           "Apartment",
    "single_family_house": "Single Family House",
    "terraced_house":      "Terraced House",
}


def unit_label(ut: str) -> str:
    """Mappt einen unit_type-Rohwert auf den Anzeigenamen (ohne Unterstriche)."""
    return UNIT_TYPE_LABELS.get(str(ut), str(ut).replace("_", " ").title())


# Englische Anzeigenamen für die Baujahr-Klassen (CSV-Inhalt bleibt unverändert)
BAUJAHR_LABELS_EN = {
    "vor 1919":  "before 1919",
    "1919–1948": "1919–1948",
    "1949–1968": "1949–1968",
    "1969–1978": "1969–1978",
    "1979–1994": "1979–1994",
    "1995–2009": "1995–2009",
    "ab 2010":   "from 2010",
    "Unbekannt": "Unknown",
}

# Jahreszeiten (Monatszuordnung) und Farben für die Tagesprofil-Plots
SEASON_MONTHS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11],
}
SEASON_COLORS = {
    "Winter": "steelblue",
    "Spring": "tab:green",
    "Summer": "gold",
    "Autumn": "darkorange",
}

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


def list_meter_files(data_dir: str) -> list[str]:
    """
    Listet Rohdaten-CSVs. Kontextdatei und selected_*-Outputs werden ausgeschlossen.
    """
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    exclude = {CONTEXT_FILENAME, os.path.basename(OUT_LONG), os.path.basename(OUT_META)}
    files = [
        f for f in files
        if os.path.basename(f) not in exclude
        and not os.path.basename(f).startswith("selected_")
    ]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"Keine Rohdaten-CSV-Dateien in '{data_dir}' gefunden."
        )
    return files


def check_columns(path: str, sep: str, required: list[str]) -> list[str]:
    """
    Liest nur den Header einer CSV und gibt fehlende Pflichtspalten zurück. [7]
    Spalten werden normalisiert (BOM, Whitespace entfernt).
    """
    raw_cols = pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
    norm_cols = [c.replace("\ufeff", "").strip() for c in raw_cols]
    missing = [c for c in required if c not in norm_cols]
    return missing


def parse_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(",", ".", regex=False)
    s2 = s2.str.replace(r"[^\d\.\-]", "", regex=True)
    s2 = s2.replace("", np.nan)
    return pd.to_numeric(s2, errors="coerce")


def load_context(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, CONTEXT_FILENAME)
    sep  = detect_csv_separator(path)
    df_ctx = pd.read_csv(path, sep=sep, low_memory=False)

    missing = [c for c in [CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION]
               if c not in df_ctx.columns]
    if missing:
        raise KeyError("Kontext-Spalten fehlen: " + ", ".join(missing))

    df_ctx = df_ctx[[CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION]].copy()
    df_ctx.columns = ["meter_id", "unit_type", "construction_year"]
    df_ctx["meter_id"]          = df_ctx["meter_id"].astype(str).str.strip()
    df_ctx["unit_type"]         = df_ctx["unit_type"].astype(str).str.strip()
    df_ctx["construction_year"] = pd.to_numeric(df_ctx["construction_year"], errors="coerce")
    return df_ctx


def baujahr_klasse(year: float) -> str:
    if pd.isna(year): return "Unbekannt"
    y = int(year)
    if y < 1919: return "vor 1919"
    if y < 1949: return "1919–1948"
    if y < 1969: return "1949–1968"
    if y < 1979: return "1969–1978"
    if y < 1995: return "1979–1994"
    if y < 2010: return "1995–2009"
    return "ab 2010"


def annual_kwh_from_energi1(df: pd.DataFrame) -> pd.Series:
    """Jahresverbrauch je Meter: max - min des kumulierten Zählerstands."""
    e = df.dropna(subset=["energi1_kwh"]).groupby("meter_id")["energi1_kwh"].agg(["min", "max"])
    return (e["max"] - e["min"]).rename("annual_kwh_2019")


def peak_kw_from_effekt1(df_hourly: pd.DataFrame) -> pd.Series:
    """
    Tatsächliche Spitzenlast je Meter aus stündlichen Effekt1-Mittelwerten. [1]
    Konsistent mit Step-1-Korrektur (kein Rückgriff mehr auf Maks.-effekt 1).
    """
    return df_hourly.max(axis=0, skipna=True).rename("peak_kw_2019")


def build_hourly_effect1(df_2019: pd.DataFrame) -> pd.DataFrame:
    """Stündliche Effekt1-Matrix: 8760 Zeilen x n_meter Spalten. NaN bleibt NaN."""
    df = df_2019[["datetime", "meter_id", "effekt1_kw"]].copy()
    df["datetime_h"] = df["datetime"].dt.floor("h")
    df_agg = df.groupby(["datetime_h", "meter_id"], as_index=False)["effekt1_kw"].mean()
    df_hourly = df_agg.pivot(index="datetime_h", columns="meter_id", values="effekt1_kw")
    full_index = pd.date_range("2019-01-01 00:00", "2019-12-31 23:00", freq="h")
    df_hourly  = df_hourly.reindex(full_index)
    df_hourly.index.name = "datetime"
    return df_hourly


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Füllt NaN-Lücken bis max_gap Stunden. Größere bleiben NaN."""
    if not series.isna().any():
        return series
    return series.interpolate(method="time", limit=max_gap, limit_direction="both")


def proportional_allocation(counts: pd.Series, n_target: int) -> pd.Series:
    """Exakt proportionale Allokation per Largest Remainder (Hamilton-Methode)."""
    weights  = counts / counts.sum()
    raw      = weights * n_target
    base     = np.floor(raw).astype(int)
    remainder = raw - base
    missing  = n_target - base.sum()
    if missing > 0:
        add_idx = remainder.sort_values(ascending=False).head(missing).index
        base.loc[add_idx] += 1
    elif missing < 0:
        sub_idx = remainder.sort_values(ascending=True).head(-missing).index
        base.loc[sub_idx] -= 1
    return base


def stratified_sample_within_type(
    df_type: pd.DataFrame,
    n_pick: int,
    n_bins: int,
    rng: np.random.Generator,
) -> tuple[pd.Index, pd.Series]:
    """
    Wählt n_pick Meter stratifiziert nach annual_kwh_2019 (Quantil-Bins).

    Korrektur [4]: Gibt sowohl die gewählten Indizes als auch die Bin-Labels
    zurück, damit Bins nur einmal berechnet werden und Meta-CSV konsistent ist.

    Returns
    -------
    picked : pd.Index   – ausgewählte meter_id
    bin_labels : pd.Series – Bin-Label je meter_id (für alle Meter im Pool)
    """
    df_type    = df_type.dropna(subset=["annual_kwh_2019"]).copy()
    bin_labels = pd.Series(pd.NA, index=df_type.index, dtype=object)

    if len(df_type) == 0 or n_pick <= 0:
        return pd.Index([]), bin_labels

    bins = min(n_bins, max(1, df_type["annual_kwh_2019"].nunique()))
    try:
        df_type["bin"] = pd.qcut(df_type["annual_kwh_2019"], q=bins, duplicates="drop")
    except ValueError:
        df_type["bin"] = "all"

    bin_labels.loc[df_type.index] = df_type["bin"].astype(str)

    bin_counts = df_type["bin"].value_counts().sort_index()
    alloc      = proportional_allocation(bin_counts, n_pick)

    picks = []
    for b, k in alloc.items():
        if k <= 0:
            continue
        pool = df_type[df_type["bin"] == b].index.to_numpy()
        if k >= len(pool):
            picks.extend(pool.tolist())
        else:
            picks.extend(rng.choice(pool, size=k, replace=False).tolist())

    return pd.Index([str(p) for p in picks]), bin_labels  # [5] str-Konsistenz


# =============================================================================
# 2) PLOT-FUNKTIONEN (identisch zu Step 1, aber für 267 Meter) [11]
# =============================================================================

def _nan_fraction_per_hour(df_hourly: pd.DataFrame) -> pd.Series:
    return df_hourly.isna().mean(axis=1) * 100


def _add_nan_axis(ax_main: plt.Axes, df_hourly: pd.DataFrame,
                  line_color: str = "red"):
    """
    Zweite Y-Achse mit Anteil fehlender Meter je Stunde.

    Achsenzahlen und Achsentitel sind schwarz; die gestrichelte Linie behält ihre
    Farbe (Standard rot, in Plot 6 schwarz wegen des bunten Hintergrunds).
    Gibt die zweite Achse und das Linien-Handle (für die Legende) zurück.
    """
    nan_frac = _nan_fraction_per_hour(df_hourly)
    ax_r = ax_main.twinx()
    ax_r.fill_between(df_hourly.index, nan_frac.values, alpha=0.12, color=line_color)
    line, = ax_r.plot(df_hourly.index, nan_frac.values, color=line_color,
                      linewidth=0.9, linestyle="--",
                      label="Share of meters without reading [%]")
    ax_r.set_ylabel("Share of meters without reading [%]", color="black", fontsize=9)
    ax_r.tick_params(axis="y", labelcolor="black")
    ax_r.set_ylim(0, 100)
    return ax_r, line


def _pie_with_outside_small_pct(ax: plt.Axes, values, labels, colors,
                                threshold: float = 8.0, startangle: int = 90,
                                radius: float = 1.0,
                                group_small_labels: bool = False):
    """
    Kreisdiagramm, bei dem Prozentwerte kleiner Segmente (< threshold %) außerhalb
    des Diagramms platziert und mit einer Linie zum Segment verbunden werden.
    Verhindert, dass kleine Prozentzahlen in benachbarte Segmente hineinragen.

    radius : Radius des Kreises (kleiner = mehr Platz für Außenbeschriftung/Titel).
    group_small_labels : Wenn True, wird für kleine Segmente die Bezeichnung
        zusammen mit der Prozentzahl als gruppierter Block außerhalb platziert
        (statt am Segment), sodass Strich, Zahl und Bezeichnung klar zusammengehören.
    """
    total = float(sum(values))
    is_small = [((v / total * 100) if total > 0 else 0.0) < threshold
                for v in values]

    if group_small_labels:
        # Bezeichnung kleiner Segmente nicht am Kreis platzieren (kommt nach außen)
        pie_labels = ["" if s else lab for lab, s in zip(labels, is_small)]
    else:
        pie_labels = list(labels)

    wedges, _ = ax.pie(values, labels=pie_labels, colors=colors,
                       startangle=startangle, radius=radius,
                       labeldistance=1.12, textprops={"fontsize": 9})

    for w, val, lab, small in zip(wedges, values, labels, is_small):
        pct = (val / total * 100) if total > 0 else 0.0
        ang = np.deg2rad((w.theta2 + w.theta1) / 2.0)
        x, y = np.cos(ang), np.sin(ang)
        if small:
            if group_small_labels:
                # Bezeichnung + Prozentzahl als gruppierter Block außerhalb,
                # mit Verbindungsstrich; Komponenten berühren sich nicht (shrinkB).
                text = f"{lab}\n{pct:.1f}%"
            else:
                text = f"{pct:.1f}%"
            # Kleine Segmente nach links bzw. rechts versetzen (Knick-Leitlinie),
            # damit sich mehrere kleine Labels nebeneinander nicht überlappen.
            ang_deg = (w.theta1 + w.theta2) / 2.0
            side = 1 if x >= 0 else -1
            ha = "left" if side > 0 else "right"
            ax.annotate(
                text,
                xy=(x * radius, y * radius),
                # horizontaler Versatz reduziert -> kürzere horizontale Strecke,
                # Beschriftung näher am Kreis; Höhe (y) bleibt für Abstand erhalten.
                xytext=(side * 0.80 * radius, y * 1.30 * radius),
                ha=ha, va="center", fontsize=9, linespacing=1.4,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8,
                                shrinkA=2, shrinkB=6,
                                connectionstyle=f"angle,angleA=0,angleB={ang_deg}"),
            )
        else:
            ax.text(0.6 * radius * x, 0.6 * radius * y, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=9)
    return wedges


def plot_annual_consumption_stats(meter_stats: pd.DataFrame, n_sample: int) -> None:
    """
    Plots 1–2: Histogramm, Scatter (Jahresverbrauch / Spitzenlast je Haustyp).
    Identisch zu Step 1, aber für die n_sample ausgewählten Meter.
    """
    unit_types_sorted = sorted(meter_stats["unit_type"].fillna("unbekannt").unique().tolist())
    palette    = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4", "#607D8B"]
    type_color = {t: palette[i % len(palette)] for i, t in enumerate(unit_types_sorted)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(left=0.07, right=0.965, top=0.82, bottom=0.12, wspace=0.28)
    fig.suptitle("Reduced Heat Demand Data for 2019 per type of house",
                 fontsize=14, fontweight="bold", y=0.96)

    # Plot 1: Histogramm Jahresverbrauch
    # Reihenfolge nach Anzahl absteigend -> kleinste Kategorie (z.B. Apartment/grün)
    # wird zuletzt und mit höchstem zorder gezeichnet und bleibt sichtbar.
    ax = axes[0]
    order_by_count = (
        meter_stats["unit_type"].value_counts()
        .reindex(unit_types_sorted).fillna(0)
        .sort_values(ascending=False).index.tolist()
    )
    for z, ut in enumerate(order_by_count):
        sub = meter_stats.loc[meter_stats["unit_type"] == ut, "annual_kwh_2019"].dropna() / 1000
        if len(sub):
            ax.hist(sub, bins=20, color=type_color[ut], alpha=1.0,
                    label=unit_label(ut), zorder=2 + z,
                    edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Annual Heat Demand [MWh/year]")
    ax.set_ylabel("Number of Meters")
    ax.set_title("Annual Heat Demand Distribution")
    ax.legend(fontsize=8)

    # Plot 2: Scatter Jahresverbrauch vs. Spitzenlast (Punkte voll ausgefüllt)
    ax = axes[1]
    for ut in unit_types_sorted:
        grp = meter_stats[meter_stats["unit_type"] == ut].dropna(
            subset=["annual_kwh_2019", "peak_kw_2019"]
        )
        if len(grp):
            ax.scatter(grp["annual_kwh_2019"] / 1000, grp["peak_kw_2019"],
                       color=type_color[ut], s=18, alpha=1.0,
                       edgecolor="none", label=unit_label(ut))
    ax.set_xlabel("Annual Heat Demand [MWh/year]")
    ax.set_ylabel("Peak Load [kW]")
    ax.set_title("Annual Heat Demand vs. Peak Load")
    ax.legend(fontsize=8)

    _save(fig, "step2_annual_consumption_stats.png")
    plt.show()


def plot_meter_type_pie(meter_stats: pd.DataFrame, n_sample: int) -> None:
    """
    Plot 3: Pie – Anzahl Meter je Haustyp (kleine Prozentwerte außerhalb).
    Eigenständige Abbildung (getrennt von Histogramm/Scatter), damit sie einzeln
    (z.B. im Bericht, neben dem Step-1-Pendant) verwendet werden kann.
    """
    unit_types_sorted = sorted(meter_stats["unit_type"].fillna("unbekannt").unique().tolist())
    palette    = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4", "#607D8B"]
    type_color = {t: palette[i % len(palette)] for i, t in enumerate(unit_types_sorted)}

    fig, ax = plt.subplots(figsize=(7, 7))
    counts = (
        meter_stats["unit_type"].fillna("unbekannt")
        .value_counts().reindex(unit_types_sorted).fillna(0).astype(int)
    )
    _pie_with_outside_small_pct(
        ax,
        counts.values.tolist(),
        [unit_label(t) for t in counts.index],
        [type_color[t] for t in counts.index],
        radius=0.72,                 # kleiner -> mehr Abstand Titel <-> Außenlabel
        group_small_labels=True,     # "Apartment" + Prozentzahl gruppiert nach außen
    )
    # pad: zusätzlicher Abstand zwischen Pie-Titel und der außenliegenden Prozentzahl
    ax.set_title("Share of Meters per Type of House", fontweight="bold", pad=18)

    _save(fig, "step2_meter_type_pie.png")
    plt.show()


def plot_series_panels(
    df_hourly: pd.DataFrame,
    df_meta: pd.DataFrame,
    n_sample: int,
) -> None:
    """
    Plots 4–8: Zeitreihen Gesamtlast, Stack unit_type, Stack Baujahr,
    7-Tage gleitender Mittelwert, Tagesprofile je Jahreszeit.
    Identisch zu Step 1, aber für die n_sample ausgewählten Meter.
    NaN-Behandlung: nansum mit roter NaN-Hilfslinie (zweite Y-Achse).
    """
    total_heat = df_hourly.sum(axis=1, min_count=1)
    nan_frac   = _nan_fraction_per_hour(df_hourly)

    # meta_sample hat meter_id als Index -> Index für Spaltenzugriff auf df_hourly nutzen
    # Aggregation nach unit_type
    heat_by_type: dict[str, pd.Series] = {}
    for ut, grp in df_meta.groupby("unit_type"):
        cols = [m for m in grp.index if m in df_hourly.columns]
        if cols:
            heat_by_type[ut] = df_hourly[cols].sum(axis=1, min_count=1)

    # Aggregation nach Baujahr-Klasse
    BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                     "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]
    heat_by_baujahr: dict[str, pd.Series] = {}
    for bk, grp in df_meta.groupby("baujahr_klasse"):
        cols = [m for m in grp.index if m in df_hourly.columns]
        if cols:
            heat_by_baujahr[bk] = df_hourly[cols].sum(axis=1, min_count=1)

    types_sorted = sorted(heat_by_type.keys())
    palette      = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4", "#607D8B"]
    color_map    = {t: palette[i % len(palette)] for i, t in enumerate(types_sorted)}
    b_palette    = ["#7B1FA2", "#C62828", "#EF6C00", "#F9A825", "#558B2F", "#0277BD", "#00838F", "#9E9E9E"]
    b_color      = {bk: b_palette[i % len(b_palette)] for i, bk in enumerate(BAUJAHR_ORDER)}

    # --- Plot 4: Gesamtlast + NaN-Anteil ---
    fig1, ax1 = plt.subplots(figsize=(16, 5), constrained_layout=True)
    line_load, = ax1.plot(df_hourly.index, total_heat.values, color="#1f77b4",
                          linewidth=0.9, label="Total heat load [kW]")
    ax1.set_title("Total Heat Load of Reduced Profile in 2019", fontweight="bold")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Heat Load [kW]")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_r1, line_nan = _add_nan_axis(ax1, df_hourly)
    # Infobox + Legende in der oberen Mitte (über dem Sommer-Tief, frei von Daten)
    ax1.annotate(
        f"NaN share total: {total_heat.isna().mean():.2%}\n"
        f"Avg. NaN meters per hour: {nan_frac.mean():.1f}%\n"
        f"Max: {np.nanmax(total_heat.values):.1f} kW  "
        f"| Avg.: {np.nanmean(total_heat.values):.1f} kW",
        xy=(0.5, 0.98), xycoords="axes fraction", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        fontsize=8,
    )
    ax1.legend([line_load, line_nan],
               [line_load.get_label(), line_nan.get_label()],
               loc="upper center", bbox_to_anchor=(0.5, 0.78), fontsize=9)
    _save(fig1, "step2_total_load.png")
    plt.show()

    # --- Plot 5: Stack unit_type + Pie Jahresenergie ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 6),
                                       gridspec_kw={"width_ratios": [3, 1.25]})
    # Manuelles Layout: größerer Abstand zwischen Stack und Pie (wspace) und
    # größerer Abstand zwischen Hauptüberschrift (y) und Diagramm-Überschriften (top).
    fig2.subplots_adjust(left=0.06, right=0.965, top=0.80, bottom=0.12, wspace=0.30)
    fig2.suptitle("Reduced Heat Load per Type of House in 2019",
                  fontsize=13, fontweight="bold", y=0.98)
    bottom = np.zeros(len(df_hourly.index))
    for t in types_sorted:
        vals = np.nan_to_num(heat_by_type[t].to_numpy(), nan=0.0)
        ax2a.fill_between(df_hourly.index, bottom, bottom + vals,
                          alpha=0.85, color=color_map[t], label=unit_label(t))
        bottom += vals
    ax_r2, line_nan2 = _add_nan_axis(ax2a, df_hourly)
    ax2a.set_xlabel("Month")
    ax2a.set_ylabel("Heat Load [kW]")
    ax2a.xaxis.set_major_locator(mdates.MonthLocator())
    ax2a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    handles, labels = ax2a.get_legend_handles_labels()
    handles.append(line_nan2)
    labels.append(line_nan2.get_label())
    ax2a.legend(handles, labels, fontsize=9, loc="upper right")

    pie_labels, pie_vals, pie_cols = [], [], []
    for t in types_sorted:
        e = float(np.nansum(heat_by_type[t].values))
        if e > 0:
            pie_labels.append(unit_label(t))
            pie_vals.append(e)
            pie_cols.append(color_map[t])
    _pie_with_outside_small_pct(
        ax2b, pie_vals, pie_labels, pie_cols,
        radius=0.85,                 # etwas größer, Abstand bleibt durch Titel-pad
        group_small_labels=True,     # kleine Segmente: Bezeichnung + % seitlich versetzt
    )
    # pad: Titel nach oben, damit über dem größeren Pie genug Abstand bleibt
    ax2b.set_title("Share of Annual Heat Demand", pad=26)
    _save(fig2, "step2_load_by_type.png")
    plt.show()

    # --- Plot 6: Stack Baujahr-Klasse + NaN-Anteil ---
    fig3, ax3 = plt.subplots(figsize=(16, 6), constrained_layout=True)
    bottom = np.zeros(len(df_hourly.index))
    for bk in BAUJAHR_ORDER:
        if bk not in heat_by_baujahr:
            continue
        vals = np.nan_to_num(heat_by_baujahr[bk].to_numpy(), nan=0.0)
        ax3.fill_between(df_hourly.index, bottom, bottom + vals,
                         alpha=0.85, color=b_color[bk],
                         label=BAUJAHR_LABELS_EN.get(bk, bk))
        bottom += vals
    # NaN-Linie schwarz, da Rot auf dem bunten Hintergrund nicht sichtbar ist
    ax_r3, line_nan3 = _add_nan_axis(ax3, df_hourly, line_color="black")
    ax3.set_title("Reduced Heat Load of 2019 per Construction Year", fontweight="bold")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Heat Load [kW]")
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    handles, labels = ax3.get_legend_handles_labels()
    handles.append(line_nan3)
    labels.append(line_nan3.get_label())
    # Legende in die obere Mitte (über das Sommer-Tief), mehrspaltig und kompakt,
    # damit sie die Stack-Flächen (Winterspitzen links/rechts) nicht verdeckt.
    ax3.legend(handles, labels, fontsize=8, loc="upper center", ncol=3,
               title="Construction Year", framealpha=0.9)
    _save(fig3, "step2_load_by_construction_year.png")
    plt.show()

    # --- Plot 7: Gesamtlast – 7-Tage gleitender Mittelwert ---
    ma_7d = total_heat.rolling(window=24 * 7, center=True, min_periods=24).mean()
    fig4, ax4 = plt.subplots(figsize=(16, 5), constrained_layout=True)
    ax4.plot(df_hourly.index, total_heat.values, color="gray", alpha=0.35,
             linewidth=0.6, label="Hourly total heat load [kW]")
    ax4.plot(df_hourly.index, ma_7d.values, color="#1f77b4", linewidth=1.6,
             label="7-day moving average [kW]")
    ax4.set_title("Total Heat Load of Reduced Profile – 7-Day Moving Average",
                  fontweight="bold")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Heat Load [kW]")
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax4.legend(fontsize=9, loc="upper right")
    _save(fig4, "step2_total_load_7day_ma.png")
    plt.show()

    # --- Plot 8: Mittleres Tagesprofil je Jahreszeit ---
    df_tmp = pd.DataFrame({
        "value": total_heat.values,
        "hour":  df_hourly.index.hour,
        "month": df_hourly.index.month,
    })
    fig5, ax5 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for season, months in SEASON_MONTHS.items():
        sub = df_tmp[df_tmp["month"].isin(months)]
        profile = sub.groupby("hour")["value"].mean()
        ax5.plot(profile.index, profile.values, color=SEASON_COLORS[season],
                 linewidth=1.8, marker="o", markersize=3, label=season)
    ax5.set_title("Average Diurnal Heat Load Profile by Season (Reduced Profile)",
                  fontweight="bold")
    ax5.set_xlabel("Hour of Day")
    ax5.set_ylabel("Average Heat Load [kW]")
    ax5.set_xticks(range(0, 24, 2))
    ax5.legend(fontsize=9)
    _save(fig5, "step2_diurnal_profiles_by_season.png")
    plt.show()


# =============================================================================
# 3) VALIDIERUNG
# =============================================================================

def validate_sample(meta_all: pd.DataFrame, meta_sample: pd.DataFrame) -> None:
    """
    Statistische Validierung: Vergleich Sample vs. Gesamtpopulation. [6]

    Ausgaben:
    - unit_type-Anteile [%] (Original vs. Sample)
    - Jahresverbrauch: Quantilvergleich (10%, 25%, 50%, 75%, 90%)
    - KS-Test auf Jahresverbrauchsverteilung (p > 0.05 = kein signifikanter Unterschied)
    """
    print("\n" + "=" * 60)
    print("VALIDIERUNG: Sample vs. Gesamtpopulation")
    print("=" * 60)

    # unit_type-Anteile
    print("\nunit_type-Anteile [%]:")
    orig_shares   = meta_all["unit_type"].value_counts(normalize=True) * 100
    sample_shares = meta_sample["unit_type"].value_counts(normalize=True) * 100
    comp = pd.DataFrame({
        "Original [%]": orig_shares,
        "Sample [%]":   sample_shares,
        "Differenz [pp]": (sample_shares - orig_shares).abs(),
    }).fillna(0).round(2)
    print(comp.to_string())

    # Jahresverbrauch: Quantilvergleich
    print("\nJahresverbrauch (MWh) – Quantilvergleich:")
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    orig_q   = (meta_all["annual_kwh_2019"].dropna() / 1000).quantile(quantiles)
    sample_q = (meta_sample["annual_kwh_2019"].dropna() / 1000).quantile(quantiles)
    qcomp = pd.DataFrame({
        "Original [MWh]": orig_q,
        "Sample [MWh]":   sample_q,
        "Differenz [MWh]": (sample_q - orig_q).abs(),
    }).round(3)
    qcomp.index = [f"P{int(q*100)}" for q in quantiles]
    print(qcomp.to_string())

    # KS-Test
    orig_vals   = meta_all["annual_kwh_2019"].dropna().values
    sample_vals = meta_sample["annual_kwh_2019"].dropna().values
    ks_stat, ks_p = scipy_stats.ks_2samp(orig_vals, sample_vals)
    print(f"\nKS-Test Jahresverbrauch (Original vs. Sample):")
    print(f"  KS-Statistik: {ks_stat:.4f}")
    print(f"  p-Wert:       {ks_p:.4f}")
    if ks_p >= 0.05:
        print("  -> Kein signifikanter Unterschied (p >= 0.05). Sample repräsentativ.")
    else:
        print("  -> WARNUNG: Signifikanter Unterschied (p < 0.05). Sample prüfen!")

    print("\nNaN-Anteil nach Interpolation (gesamt):", end=" ")


# =============================================================================
# 4) MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Step 2: Auswahl 267 repräsentative Profile (stratifiziert)")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)

    # --- Dateien einlesen ---
    files = list_meter_files(DATA_DIR)
    print(f"CSV-Dateien (Smart Meter): {len(files)}")

    df_ctx = load_context(DATA_DIR)

    # Pflicht-Spalten für Rohdaten [7]
    REQUIRED_COLS = [COL_DATETIME, COL_METER_ID, COL_EFFEKT1, COL_ENERGI1]

    chunks       = []
    skipped      = []
    for i, f in enumerate(files, 1):
        sep = detect_csv_separator(f)
        print(f"  Lese {i:02d}/{len(files)}: {os.path.basename(f)} (sep='{sep}')")

        # Spaltenprüfung vor dem Einlesen [7]
        missing_cols = check_columns(f, sep, REQUIRED_COLS)
        if missing_cols:
            print(f"  WARNUNG: Datei übersprungen – fehlende Spalten: {missing_cols}")
            skipped.append({"file": os.path.basename(f), "missing": missing_cols})
            continue

        d_full = pd.read_csv(f, sep=sep, low_memory=False)
        rename_map = {c: c.replace("\ufeff", "").strip() for c in d_full.columns}
        d_full = d_full.rename(columns=rename_map)

        d = d_full[[COL_DATETIME, COL_METER_ID, COL_EFFEKT1, COL_ENERGI1]].copy()
        d = d.rename(columns={
            COL_DATETIME: "datetime",
            COL_METER_ID: "meter_id",
            COL_EFFEKT1:  "effekt1_kw",
            COL_ENERGI1:  "energi1_kwh",
        })

        d["meter_id"]    = d["meter_id"].astype(str).str.strip()
        d["datetime"]    = pd.to_datetime(d["datetime"], errors="coerce", dayfirst=DAYFIRST)
        d["effekt1_kw"]  = parse_numeric_series(d["effekt1_kw"])
        d["energi1_kwh"] = parse_numeric_series(d["energi1_kwh"])

        d = d.dropna(subset=["datetime", "meter_id"])
        d = d[d["datetime"].dt.year == 2019]
        chunks.append(d)

    if skipped:
        print(f"\nÜbersprungene Dateien (fehlende Spalten): {len(skipped)}")
        for s in skipped:
            print(f"  {s['file']}: {s['missing']}")

    df_2019 = pd.concat(chunks, ignore_index=True)
    print(f"\nZeilen 2019: {len(df_2019):,}")
    print(f"Meter 2019:  {df_2019['meter_id'].nunique():,}")

    # --- Stündliche Matrix (für Abdeckungscheck + Spitzenlast) ---
    print("\nBaue stündliche Effekt1-Matrix ...")
    df_hourly_all = build_hourly_effect1(df_2019)
    print(f"  Matrix: {df_hourly_all.shape[0]} h x {df_hourly_all.shape[1]} Meter")

    # --- Meta-Tabelle aufbauen ---
    annual   = annual_kwh_from_energi1(df_2019)
    peak_all = peak_kw_from_effekt1(df_hourly_all)

    meta = (
        pd.DataFrame({"meter_id": df_2019["meter_id"].unique()})
        .merge(df_ctx, on="meter_id", how="left")
        .set_index("meter_id")
        .join(annual, how="left")
        .join(peak_all, how="left")
    )
    meta["unit_type"] = meta["unit_type"].fillna("unbekannt")

    # Stündliche Abdeckung je Meter (Anteil Stunden mit Effekt1-Wert) [2]
    coverage = df_hourly_all.notna().sum(axis=0) / 8760
    meta     = meta.join(coverage.rename("coverage_frac"), how="left")

    # --- Filter 1: "unclear" unit_type entfernen [8] ---
    mask_unclear = meta["unit_type"].str.lower().isin(
        {u.lower() for u in EXCLUDE_UNIT_TYPES}
    )
    n_unclear = int(mask_unclear.sum())
    if n_unclear > 0:
        unclear_ids = meta[mask_unclear].index.tolist()
        print(f"\nWARNUNG: {n_unclear} Meter mit unit_type 'unclear' werden ausgeschlossen:")
        for mid in unclear_ids:
            print(f"  meter_id={mid}  unit_type={meta.loc[mid, 'unit_type']}")
        meta = meta[~mask_unclear]
    else:
        print("\nKeine 'unclear'-Meter gefunden.")

    # --- Filter 2: Mindestabdeckung Effekt 1 [2] ---
    mask_low = meta["coverage_frac"] < MIN_COVERAGE_FRAC
    n_low    = int(mask_low.sum())
    if n_low > 0:
        print(f"\nWARNUNG: {n_low} Meter mit Effekt1-Abdeckung < {MIN_COVERAGE_FRAC:.0%} "
              f"({MIN_COVERAGE_HOURS}h) ausgeschlossen.")
        print(f"  Verbleibende Meter nach Filter: {int((~mask_low).sum())}")
        meta = meta[~mask_low]
    else:
        print(f"\nAlle Meter erfüllen Mindestabdeckung ({MIN_COVERAGE_FRAC:.0%}).")

    print(f"\nPool nach Filtern: {len(meta)} Meter")

    # --- Proportionale Allokation nach unit_type ---
    type_counts  = meta["unit_type"].value_counts()
    alloc_types  = proportional_allocation(type_counts, N_TARGET)

    print("\nAllokation nach unit_type (proportional):")
    for ut, n in alloc_types.items():
        pool_size = int(type_counts.get(ut, 0))
        warn = " !! WARNUNG: Weniger Meter im Pool als benötigt" if pool_size < n else ""
        print(f"  {ut}: {n} (Pool: {pool_size}){warn}")
    print(f"  Summe: {int(alloc_types.sum())}")

    # --- Stratifizierte Auswahl [4] ---
    meta["chosen"]    = False
    meta["bin"]       = pd.NA
    chosen_list: list[str] = []

    for ut, n_pick in alloc_types.items():
        pool = meta[meta["unit_type"] == ut]
        picked_idx, bin_labels = stratified_sample_within_type(
            pool, int(n_pick), N_BINS, rng
        )
        # Bin-Labels für ALLE Meter dieser Gruppe ins Meta schreiben [4]
        valid_bin_idx = bin_labels.index[bin_labels.notna()]
        meta.loc[valid_bin_idx, "bin"] = bin_labels[valid_bin_idx]
        # Auswahl markieren
        meta.loc[picked_idx, "chosen"] = True
        chosen_list.extend([str(p) for p in picked_idx])  # [5] str-Konsistenz

    chosen = pd.Index(chosen_list).unique()

    # Fallback: exakt N_TARGET sicherstellen [5]
    if len(chosen) != N_TARGET:
        delta = N_TARGET - len(chosen)
        if delta > 0:
            rest = meta[~meta["chosen"]].dropna(subset=["annual_kwh_2019"]).index
            rest_str = pd.Index([str(r) for r in rest])
            add = rng.choice(rest_str.to_numpy(), size=delta, replace=False)
            add = pd.Index([str(a) for a in add])
            meta.loc[add, "chosen"] = True
            chosen = pd.Index([str(c) for c in chosen] + list(add)).unique()
        else:
            chosen = pd.Index(
                [str(c) for c in rng.choice(chosen.to_numpy(), size=N_TARGET, replace=False)]
            )
            meta["chosen"] = False
            meta.loc[chosen, "chosen"] = True

    print(f"\nAusgewählte Meter: {len(chosen)}")

    # --- Stündliche Profile der 267 Meter ---
    chosen_in_hourly = [m for m in chosen if m in df_hourly_all.columns]
    n_missing_hourly = len(chosen) - len(chosen_in_hourly)
    if n_missing_hourly > 0:  # [3]
        print(f"\nWARNUNG: {n_missing_hourly} ausgewählte Meter haben keine "
              f"Effekt1-Stundendaten und fehlen im Output!")

    df_sel = df_hourly_all.reindex(columns=chosen_in_hourly)

    # Interpolation kurzer Lücken
    print(f"\nInterpolation kurzer Lücken (max {INTERP_MAX_GAP_HOURS}h) ...")
    df_sel = pd.DataFrame(
        {col: interpolate_short_gaps(df_sel[col], INTERP_MAX_GAP_HOURS)
         for col in df_sel.columns},
        index=df_sel.index,
    )

    # --- Meta der ausgewählten Meter ---
    meta_sample = meta[meta["chosen"]].copy()
    meta_sample["baujahr_klasse"] = meta_sample["construction_year"].apply(baujahr_klasse)

    # Spitzenlast für die 267 Meter aus deren stündlichen Daten [1]
    meta_sample["peak_kw_2019"] = peak_kw_from_effekt1(df_sel)

    # --- Validierung [6] ---
    validate_sample(meta, meta_sample)
    nan_overall = float(df_sel.isna().mean().mean())
    print(f"{nan_overall:.4f}")

    # --- Output Long-CSV [9][10] ---
    # Jahresverbrauch und unit_type je Meter in den Output
    meter_info = meta_sample[["unit_type", "annual_kwh_2019"]].copy()

    out_long = (
        df_sel
        .reset_index()
        .melt(id_vars=["datetime"], var_name="meter_id", value_name="effekt1_kw")
    )
    out_long["meter_id"] = out_long["meter_id"].astype(str)
    out_long = out_long.merge(meter_info, left_on="meter_id", right_index=True, how="left")

    # Sortierung nach Jahresverbrauch absteigend [10]
    sort_order = (
        meta_sample["annual_kwh_2019"]
        .sort_values(ascending=False)
        .index
    )
    out_long["meter_id"] = pd.Categorical(out_long["meter_id"],
                                           categories=sort_order, ordered=True)
    out_long = out_long.sort_values(["meter_id", "datetime"]).reset_index(drop=True)
    out_long["meter_id"] = out_long["meter_id"].astype(str)

    # Spaltenreihenfolge [9]
    out_long = out_long[["datetime", "meter_id", "unit_type", "annual_kwh_2019", "effekt1_kw"]]
    out_long.to_csv(OUT_LONG, index=False)

    # --- Output Wide-CSV: Datum + Smart-Meter 1..267 ---
    # Spalten = Smart-Meter, sortiert nach Jahresverbrauch absteigend und
    # durchnummeriert (1 = höchster Jahresverbrauch, 267 = niedrigster).
    # Jede Zeile = eine Stunde des Jahres, Werte = stündliche kW.
    # Keine echte meter_id / unit_type / annual_kwh.
    wide_order = [m for m in sort_order if m in df_sel.columns]
    df_wide = df_sel.reindex(columns=wide_order).copy()
    df_wide.columns = range(1, len(wide_order) + 1)
    df_wide = df_wide.reset_index().rename(columns={"datetime": "Datum"})
    df_wide.to_csv(OUT_WIDE, index=False)

    # --- Output Meta-CSV [9][10] ---
    meta_out = (
        meta_sample
        .reset_index()
        .rename(columns={"index": "meter_id"})
        .sort_values("annual_kwh_2019", ascending=False)  # [10]
        .reset_index(drop=True)
    )
    # Spaltenreihenfolge: wichtigste Felder zuerst [9]
    cols_order = ["meter_id", "unit_type", "construction_year", "baujahr_klasse",
                  "annual_kwh_2019", "peak_kw_2019", "coverage_frac", "chosen", "bin"]
    cols_order = [c for c in cols_order if c in meta_out.columns]
    meta_out   = meta_out[cols_order]
    meta_out.to_csv(OUT_META, index=False)

    print("\nGespeichert:")
    print(f"  Profile (long): {OUT_LONG}")
    print(f"  Profile (wide): {OUT_WIDE}")
    print(f"  Meta:           {OUT_META}")

    # --- Plots für die 267 Meter [11] ---
    print("\nErstelle Plots für die 267 ausgewählten Meter ...")
    plot_annual_consumption_stats(meta_sample, len(chosen_in_hourly))
    plot_meter_type_pie(meta_sample, len(chosen_in_hourly))
    plot_series_panels(df_sel, meta_sample, len(chosen_in_hourly))


if __name__ == "__main__":
    main()
