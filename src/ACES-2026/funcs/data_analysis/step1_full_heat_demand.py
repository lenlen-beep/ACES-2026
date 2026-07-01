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

# Plots in einem eigenen Fenster öffnen (nicht inline). Ein interaktives
# GUI-Backend wird vor dem pyplot-Import gesetzt; das erste verfügbare Backend
# wird verwendet (identisch zu Step 2).
import matplotlib
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

# =============================================================================
# 0) KONFIGURATION – hier anpassen
# =============================================================================

# Pfad robust relativ zur Skriptdatei auflösen, damit das Skript unabhängig
# vom Arbeitsverzeichnis läuft (z. B. Spyder %runfile --wdir setzt das CWD auf
# den Skript-Ordner, nicht auf die Projekt-Root).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "..", "Data", "Aalborg_smart_meter_data")
)
CONTEXT_FILENAME = "contextual_data.csv"

# Cache der berechneten Plot-Daten (stündliche Matrix + Meter-Stats + Meta).
# Wird am Ende von main() geschrieben, damit show_step1_plots.py die Plots ohne
# erneutes Einlesen der ~25 Rohdateien sofort wieder anzeigen kann.
CACHE_FILE = os.path.join(_SCRIPT_DIR, "step1_plot_cache.pkl")

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

# Plot-Style (identisch zu Step 2)
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
    "non_residential":     "Non-Residential",
    "unclear":             "Unclear",
    "unbekannt":           "Unknown",
}


def unit_label(ut: str) -> str:
    """Mappt einen unit_type-Rohwert auf den Anzeigenamen (ohne Unterstriche)."""
    return UNIT_TYPE_LABELS.get(str(ut), str(ut).replace("_", " ").title())


# Feste Farbe je unit_type (verankert nach Kategorie-NAME, nicht nach Position).
# Dadurch sind die Farben unabhängig davon, welche Kategorien im Datensatz
# vorkommen oder wie sie sortiert sind – Step 1 und Step 2 zeigen dieselbe
# Kategorie immer in derselben Farbe.
UNIT_TYPE_COLORS = {
    "apartment":           "#4CAF50",  # grün
    "single_family_house": "#2196F3",  # blau
    "terraced_house":      "#FF9800",  # orange
    "non_residential":     "#9C27B0",  # violett
    "unclear":             "#E91E63",  # pink
    "unbekannt":           "#607D8B",  # grau
}
# Fallback für evtl. weitere, hier nicht gelistete unit_types
UNIT_TYPE_FALLBACK = ["#00BCD4", "#795548", "#3F51B5", "#CDDC39"]


def type_colors(unit_types) -> dict:
    """Farbe je unit_type: feste Zuordnung (UNIT_TYPE_COLORS), sonst Fallback."""
    cmap, k = {}, 0
    for t in unit_types:
        if t in UNIT_TYPE_COLORS:
            cmap[t] = UNIT_TYPE_COLORS[t]
        else:
            cmap[t] = UNIT_TYPE_FALLBACK[k % len(UNIT_TYPE_FALLBACK)]
            k += 1
    return cmap


# Englische Anzeigenamen für die Baujahr-Klassen (Datenwerte bleiben unverändert)
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
    """
    Plots 1–3 (Stil identisch zu Step 2, aber für den VOLLEN Datensatz):

    Plot 1 – Histogramm Jahresverbrauch (MWh) je unit_type
    Plot 2 – Scatter Jahresverbrauch vs. Spitzenlast je unit_type
    Plot 3 – Pie: ANZAHL Meter je unit_type (Haustyp-Struktur, nicht Energiebeitrag)

    Inhaltlich identisch zur Vorversion; nur Stil/Abstände/Beschriftung wie Step 2.
    """
    unit_types_sorted = sorted(meter_stats["unit_type"].fillna("unbekannt").unique().tolist())
    type_color = type_colors(unit_types_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Manuelles Layout: größerer Abstand zwischen den Diagrammen (wspace) und
    # größerer Abstand zwischen Hauptüberschrift (y) und Diagramm-Überschriften (top).
    fig.subplots_adjust(left=0.055, right=0.965, top=0.80, bottom=0.12, wspace=0.34)
    fig.suptitle("Full Heat Demand Data for 2019 per Type of House",
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

    # Plot 3: Pie – Anzahl Meter je Haustyp (kleine Prozentwerte außerhalb)
    ax = axes[2]
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
    ax.set_title("Share of Meters per Type of House", pad=18)

    plt.show()


def _nan_fraction_per_hour(df_hourly: pd.DataFrame) -> pd.Series:
    """Anteil NaN-Meter je Stunde in Prozent (0% = alle vorhanden, 100% = alle NaN)."""
    return df_hourly.isna().mean(axis=1) * 100


def _add_nan_axis(ax_main: plt.Axes, df_hourly: pd.DataFrame,
                  line_color: str = "red"):
    """
    Zweite Y-Achse mit Anteil fehlender Meter je Stunde (identisch zu Step 2).

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


def _place_outside_labels(ax: plt.Axes, entries, side: int, radius: float,
                          min_gap_frac: float = 0.36) -> None:
    """
    Verteilt Außen-Labels einer Pie-Seite (links oder rechts) vertikal so, dass
    zwischen ihnen ein Mindestabstand bleibt (keine Überlappung), und verbindet
    jedes Label per Leitlinie mit seinem Segment.

    entries : Liste von (cy, cx, text) – cx/cy = Einheits-Mittelwinkel des Segments.
    side    : +1 = rechte Halbebene, -1 = linke Halbebene.
    """
    if not entries:
        return
    # Von oben nach unten sortieren
    entries = sorted(entries, key=lambda e: e[0], reverse=True)
    min_gap = min_gap_frac * radius
    desired = [cy * radius for cy, _, _ in entries]
    ys = list(desired)
    # Mindestabstand nach unten erzwingen (verhindert Überlappen)
    for i in range(1, len(ys)):
        if ys[i] > ys[i - 1] - min_gap:
            ys[i] = ys[i - 1] - min_gap
    # Label-Block auf die gewünschten Positionen zentrieren
    shift = (sum(desired) - sum(ys)) / len(ys)
    ys = [y + shift for y in ys]
    # In den sichtbaren Bereich klemmen
    ylim = 1.45 * radius
    if ys[0] > ylim:
        ys = [y - (ys[0] - ylim) for y in ys]
    if ys[-1] < -ylim:
        ys = [y + (-ylim - ys[-1]) for y in ys]

    x_text = side * 1.22 * radius
    ha = "left" if side > 0 else "right"
    for (cy, cx, text), yt in zip(entries, ys):
        ax.annotate(
            text,
            xy=(cx * radius, cy * radius),          # Segment-Rand
            xytext=(x_text, yt),                     # entzerrte Label-Position
            ha=ha, va="center", fontsize=9, linespacing=1.4,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8,
                            shrinkA=2, shrinkB=4,
                            connectionstyle="arc3,rad=0.0"),
        )


def _pie_with_outside_small_pct(ax: plt.Axes, values, labels, colors,
                                threshold: float = 8.0, startangle: int = 90,
                                radius: float = 1.0,
                                group_small_labels: bool = True):
    """
    Kreisdiagramm mit außenliegenden Beschriftungen und garantiert
    überlappungsfreien Labels (vertikale Entzerrung je Seite, siehe
    _place_outside_labels):

    - Große Segmente (>= threshold %): Prozentwert mittig im Segment, Name außen.
    - Kleine Segmente (< threshold %): Name + Prozentwert gemeinsam außen.

    Jede Außenbeschriftung ist über eine Leitlinie mit ihrem Segment verbunden;
    Linien und Texte berühren oder überlappen sich nicht.

    radius : Radius des Kreises (kleiner = mehr Platz für Außenbeschriftung/Titel).
    group_small_labels : aus Kompatibilität erhalten, ohne Wirkung (Namen liegen
        grundsätzlich außen).
    """
    total = float(sum(values)) or 1.0
    wedges, _ = ax.pie(values, colors=colors, startangle=startangle,
                       radius=radius, textprops={"fontsize": 9})

    # Außen-Labels sammeln, getrennt nach rechter/linker Halbebene
    right, left = [], []
    for w, val, lab in zip(wedges, values, labels):
        pct = val / total * 100.0
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        cx, cy = np.cos(ang), np.sin(ang)
        if pct < threshold:
            text = f"{lab}\n{pct:.1f}%"
        else:
            # Prozentwert mittig ins Segment
            ax.text(0.6 * radius * cx, 0.6 * radius * cy, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=9)
            text = f"{lab}"
        (right if cx >= 0 else left).append((cy, cx, text))

    _place_outside_labels(ax, right, side=+1, radius=radius)
    _place_outside_labels(ax, left,  side=-1, radius=radius)
    return wedges


def plot_series_panels(df_hourly: pd.DataFrame, df_meta: pd.DataFrame) -> None:
    """
    Plots 4–6 (Stil identisch zu Step 2, aber für den VOLLEN Datensatz):
    Zeitreihe Gesamtlast, Stack nach unit_type + Pie Jahresenergie, Stack nach Baujahr.

    NaN-Behandlung (inhaltlich unverändert):
    - Stündliche Summen mit nansum (min_count=1): Fehlende Meter tragen 0 bei,
      vollständig fehlende Stunden bleiben NaN.
    - Jede Figur zeigt eine zweite Y-Achse (gestrichelt) mit dem prozentualen
      Anteil fehlender Meter je Stunde -> hohe Werte mit Vorsicht interpretieren.
    """
    total_heat = df_hourly.sum(axis=1, min_count=1)
    nan_frac   = _nan_fraction_per_hour(df_hourly)

    # Aggregation nach unit_type (meter_id ist hier eine Spalte von df_meta)
    heat_by_type: dict[str, pd.Series] = {}
    for ut, mids in df_meta.groupby("unit_type")["meter_id"]:
        cols = [m for m in mids if m in df_hourly.columns]
        if cols:
            heat_by_type[ut] = df_hourly[cols].sum(axis=1, min_count=1)

    # Aggregation nach Baujahr-Klasse
    BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                     "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]
    heat_by_baujahr: dict[str, pd.Series] = {}
    for bk, mids in df_meta.groupby("baujahr_klasse")["meter_id"]:
        cols = [m for m in mids if m in df_hourly.columns]
        if cols:
            heat_by_baujahr[bk] = df_hourly[cols].sum(axis=1, min_count=1)

    types_sorted = sorted(heat_by_type.keys())
    color_map    = type_colors(types_sorted)
    b_palette    = ["#7B1FA2", "#C62828", "#EF6C00", "#F9A825", "#558B2F", "#0277BD", "#00838F", "#9E9E9E"]
    b_color      = {bk: b_palette[i % len(b_palette)] for i, bk in enumerate(BAUJAHR_ORDER)}

    # --- Plot 4: Gesamtlast + NaN-Anteil ---
    fig1, ax1 = plt.subplots(figsize=(16, 5), constrained_layout=True)
    line_load, = ax1.plot(df_hourly.index, total_heat.values, color="#1f77b4",
                          linewidth=0.9, label="Total heat load [kW]")
    ax1.set_title("Total Heat Load in 2019", fontweight="bold")
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
    plt.show()

    # --- Plot 5: Stack unit_type + Pie Jahresenergie ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 6),
                                       gridspec_kw={"width_ratios": [3, 1.25]})
    # Manuelles Layout: größerer Abstand zwischen Stack und Pie (wspace) und
    # größerer Abstand zwischen Hauptüberschrift (y) und Diagramm-Überschriften (top).
    fig2.subplots_adjust(left=0.06, right=0.965, top=0.80, bottom=0.12, wspace=0.30)
    fig2.suptitle("Full Heat Load per Type of House in 2019",
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

    # Pie: ENERGIEBEITRAG je Haustyp (inhaltlich verschieden von Plot 3: Meter-Anzahl)
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
    ax3.set_title("Full Heat Load of 2019 per Construction Year", fontweight="bold")
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

    # Cache der Plot-Daten schreiben, damit show_step1_plots.py die Plots ohne
    # erneutes Einlesen der Rohdaten sofort wieder anzeigen kann.
    print(f"\nSchreibe Plot-Cache: {CACHE_FILE}")
    pd.to_pickle(
        {"df_hourly": df_hourly, "meter_stats": meter_stats, "df_meta": df_meta},
        CACHE_FILE,
    )

    # Plots 1–3: Statistik-Panel
    plot_stats_panel(meter_stats)

    # Plots 4–6: Zeitreihen mit NaN-Transparenz
    plot_series_panels(df_hourly, df_meta)


if __name__ == "__main__":
    main()
