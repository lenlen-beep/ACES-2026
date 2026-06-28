"""ACES Projekt 2026 – Step 1 (bereinigt)

Ziel (Step 1):
- Smart-Meter-CSV (25 Dateien) + Kontextdaten (1 CSV im selben Ordner) einlesen
- Nur Jahr 2019, nur relevante Spalten: Zeitstempel, MeterID, Effekt 1 (kW)
- Stündliche Zeitreihe je Meter erstellen (Index = alle 8760 Stunden in 2019)
- Fehlende Stunden zunächst als NaN belassen (KEINE Interpolation in Step 1)
- Plots:
  1) Gesamtwärmebedarf (Summe über alle Meter) 2019
  2) Gesamtwärmebedarf nach Haustyp (unit_type) + Kuchendiagramm Anteile
  3) Gesamtwärmebedarf nach Baujahr-Klasse (construction_year)

Warum diese Bereinigung:
- Performance: Nur benötigte Spalten lesen + vektorisierte Aggregation (groupby/pivot)
- Robustheit: Separator automatisch erkennen, Kontextdatei sicher ausschließen
- Nachvollziehbarkeit: Checks für Datei-Zeiträume, Abdeckung 2019, Duplikate pro Stunde

Hinweise:
- Passe im Konfig-Block unten nur die Dateinamen/Spaltennamen an.
- pandas>=2.x: Frequenz bitte "h" statt "H".
"""

from __future__ import annotations

import os
import glob
import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================================
# 0) KONFIGURATION – hier anpassen
# =============================================================================

DATA_DIR = r"/Users/nele/Documents/GitHub/ACES-2026/src/ACES-2026/Data/Aalborg_smart_meter_data"  # <-- ANPASSEN
CONTEXT_FILENAME = "contextual_data.csv"  # <-- ANPASSEN falls Name abweicht

# Smart-Meter Spalten (exakt wie in deinen CSVs)
COL_DATETIME = "RoundedReadTime"          # alternativ: "Aflæsningstidspunkt"
COL_METER_ID = "MeterID"
COL_EFFEKT1  = "Effekt 1"                 # wichtig: Leerzeichen!

# Kontextdaten Spalten
CTX_METER_ID     = "meter_id"
CTX_UNIT_TYPE    = "unit_type"
CTX_CONSTRUCTION = "construction_year"         # ggf. z.B. "construction_year" anpassen

# Datums-Parsing
DAYFIRST = True  # bei europäischen Formaten meist richtig

# Output
OUTPUT_DIR = os.path.join(DATA_DIR, "plots_step1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot-Style
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# =============================================================================
# 1) Hilfsfunktionen
# =============================================================================

def detect_csv_separator(path: str, default: str = ",") -> str:
    """Erkennt das Trennzeichen aus einem Text-Sample.

    Warum: Manche Exporte sind ';' statt ','. Wenn sep falsch ist, passen usecols nicht.
    """
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        return default


def list_smart_meter_files(data_dir: str, context_filename: str) -> list[str]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    files = [f for f in files if os.path.basename(f) != context_filename]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"Keine Smart-Meter-CSV-Dateien in '{data_dir}' gefunden (außer '{context_filename}')."
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


def read_smart_meter_data(files: list[str]) -> pd.DataFrame:
    """Liest alle Smart-Meter-Dateien ein – nur benötigte Spalten."""
    chunks = []
    for i, f in enumerate(files, 1):
        sep = detect_csv_separator(f)
        print(f"  Lese {i:02d}/{len(files)}: {os.path.basename(f)} (sep='{sep}')")
        chunks.append(
            pd.read_csv(
                f,
                sep=sep,
                usecols=[COL_DATETIME, COL_METER_ID, COL_EFFEKT1],
                low_memory=False,
            )
        )
    df_raw = pd.concat(chunks, ignore_index=True)
    return df_raw


def parse_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Spalten vereinheitlichen und Datentypen bereinigen."""
    df = df_raw[[COL_DATETIME, COL_METER_ID, COL_EFFEKT1]].copy()
    df.columns = ["datetime", "meter_id", "effekt1_kw"]

    # datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=DAYFIRST)

    # Effekt 1: manchmal steht "2.9 kW" o.ä. -> Nicht-Ziffern entfernen
    df["effekt1_kw"] = (
        df["effekt1_kw"]
        .astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    # IDs als String (verhindert Probleme mit führenden Nullen / mixed types)
    df["meter_id"] = df["meter_id"].astype(str)

    n_before = len(df)
    df = df.dropna(subset=["datetime", "meter_id"])
    print(f"  Entfernt (NaT/fehlende ID): {n_before - len(df):,} Zeilen")
    return df


def load_context(data_dir: str, context_filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, context_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Kontextdatei nicht gefunden: {path}")
    sep = detect_csv_separator(path)
    print(f"Lese Kontextdaten: {context_filename} (sep='{sep}')")
    df_ctx = pd.read_csv(path, sep=sep, low_memory=False)
    print(f"  Kontext-Spalten: {list(df_ctx.columns)}")

    # benötigte Spalten
    missing = [c for c in [CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION] if c not in df_ctx.columns]
    if missing:
        raise KeyError(
            "Kontext-Spalten fehlen: " + ", ".join(missing) +
            "\nBitte CTX_* Variablen oben an die echten Spaltennamen anpassen."
        )

    df_ctx = df_ctx[[CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION]].copy()
    df_ctx.columns = ["meter_id", "unit_type", "construction_year"]
    df_ctx["meter_id"] = df_ctx["meter_id"].astype(str)
    df_ctx["construction_year"] = pd.to_numeric(df_ctx["construction_year"], errors="coerce")
    df_ctx["unit_type"] = df_ctx["unit_type"].astype(str)
    return df_ctx


def check_2019_coverage(df_2019: pd.DataFrame) -> None:
    """Checks: fehlende Stunden je Meter + Duplikate pro Stunde."""
    df_2019 = df_2019.copy()
    df_2019["datetime_h"] = df_2019["datetime"].dt.floor("h")

    expected_hours = 8760
    hours_per_meter = df_2019.groupby("meter_id")["datetime_h"].nunique()
    missing_hours = expected_hours - hours_per_meter

    print("\nCHECK B1: Stunden-Abdeckung 2019 je Meter")
    print(f"  Meter insgesamt (mit Daten in 2019): {hours_per_meter.size}")
    print(f"  Meter mit vollständigen 8760h:       {(missing_hours == 0).sum()}")
    print(f"  Meter mit fehlenden Stunden:         {(missing_hours > 0).sum()}")
    print(f"  Max fehlende Stunden (worst case):   {missing_hours.max()}")
    print("  Top 10 fehlende Stunden:")
    print(missing_hours.sort_values(ascending=False).head(10))

    counts_per_hour = df_2019.groupby(["meter_id", "datetime_h"]).size()
    print("\nCHECK B2: Mehrfachmessungen pro Stunde?")
    print(counts_per_hour.describe())
    print("  Anteil Stunden mit >1 Messung:", float((counts_per_hour > 1).mean()))


def build_hourly_matrix_2019(df_2019: pd.DataFrame) -> pd.DataFrame:
    """Vektorisiert: stündliche Matrix (8760 x n_meter), fehlende Werte = NaN."""
    # auf Stunde runden
    df_2019 = df_2019.copy()
    df_2019["datetime_h"] = df_2019["datetime"].dt.floor("h")

    # falls mehrere Messungen pro Stunde: Mittelwert
    df_agg = (
        df_2019
        .groupby(["datetime_h", "meter_id"], as_index=False)["effekt1_kw"]
        .mean()
    )

    # Pivot -> Stunden x Meter
    df_hourly = df_agg.pivot(index="datetime_h", columns="meter_id", values="effekt1_kw")

    # vollständiger Index 2019
    full_index = pd.date_range("2019-01-01 00:00", "2019-12-31 23:00", freq="h")
    df_hourly = df_hourly.reindex(full_index)
    df_hourly.index.name = "datetime"

    return df_hourly


def baujahr_klasse(year: float) -> str:
    if pd.isna(year):
        return "Unbekannt"
    y = int(year)
    if y < 1919:
        return "vor 1919"
    if y < 1949:
        return "1919–1948"
    if y < 1969:
        return "1949–1968"
    if y < 1979:
        return "1969–1978"
    if y < 1995:
        return "1979–1994"
    if y < 2010:
        return "1995–2009"
    return "ab 2010"


# =============================================================================
# 2) MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ACES – Step 1: Wärmelast 2019 je Smart-Meter (ohne Interpolation)")
    print("=" * 60)

    # Dateien finden
    files = list_smart_meter_files(DATA_DIR, CONTEXT_FILENAME)
    print(f"Gefundene Smart-Meter-Dateien: {len(files)}")

    # CHECK A: Datei-Zeiträume (hilft bei der Frage 'Januar wo?')
    print("\nCHECK A: Datei-Zeitabdeckung (min/max pro Datei)")
    cov = quick_file_time_coverage(files)
    print(cov.to_string(index=False))

    # Smart-Meter Daten lesen
    print("\nSchritt 1: Einlesen (nur 3 Spalten) ...")
    df_raw = read_smart_meter_data(files)
    print(f"  Roh geladen: {len(df_raw):,} Zeilen")

    # Clean + Parse
    print("\nSchritt 2: Bereinigung & Parsing ...")
    df = parse_and_clean(df_raw)
    print(f"  Nach Bereinigung: {len(df):,} Zeilen")
    print(f"  datetime min/max: {df['datetime'].min()} .. {df['datetime'].max()}")
    print(f"  unique meter_id:  {df['meter_id'].nunique():,}")

    # Filter 2019
    print("\nSchritt 3: Filter auf Jahr 2019 ...")
    df_2019 = df[df["datetime"].dt.year == 2019].copy()
    print(f"  Zeilen 2019:      {len(df_2019):,}")
    print(f"  unique meter 2019:{df_2019['meter_id'].nunique():,}")
    if len(df_2019) == 0:
        raise ValueError("df_2019 ist leer – bitte COL_DATETIME/DAYFIRST prüfen.")

    # Kontext laden + join
    print("\nSchritt 4: Kontextdaten laden & verknüpfen ...")
    df_ctx = load_context(DATA_DIR, CONTEXT_FILENAME)

    # Meta (ein Eintrag pro Meter)
    df_meta = (
        df_2019[["meter_id"]]
        .drop_duplicates()
        .merge(df_ctx, on="meter_id", how="left")
    )
    n_missing_ctx = df_meta["unit_type"].isna().sum()
    if n_missing_ctx:
        print(f"  Warnung: {n_missing_ctx} Meter ohne Kontextdaten (unit_type NaN)")
        df_meta["unit_type"] = df_meta["unit_type"].fillna("unbekannt")
    df_meta["baujahr_klasse"] = df_meta["construction_year"].apply(baujahr_klasse)

    # Checks zu Abdeckung + Duplikaten
    check_2019_coverage(df_2019)

    # Stündliche Matrix bauen (NaN bleiben NaN)
    print("\nSchritt 5: Stündliche Matrix (8760h) bauen – NaNs bleiben NaN ...")
    df_hourly = build_hourly_matrix_2019(df_2019)
    print(f"  Matrix: {df_hourly.shape[0]} Stunden x {df_hourly.shape[1]} Meter")

    # --- Aggregationen für Plots ---
    # Gesamt (wenn alles NaN in einer Stunde: Ergebnis NaN)
    total_heat = df_hourly.sum(axis=1, min_count=1)

    # Nach unit_type
    heat_by_type = {}
    for ut, mids in df_meta.groupby("unit_type")["meter_id"]:
        mids = [m for m in mids if m in df_hourly.columns]
        if mids:
            heat_by_type[ut] = df_hourly[mids].sum(axis=1, min_count=1)

    # Nach Baujahr-Klasse
    BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                     "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]
    heat_by_baujahr = {}
    for bk, mids in df_meta.groupby("baujahr_klasse")["meter_id"]:
        mids = [m for m in mids if m in df_hourly.columns]
        if mids:
            heat_by_baujahr[bk] = df_hourly[mids].sum(axis=1, min_count=1)

    # =============================================================================
    # Zusätzliche Auswertung (wie Fremd-Code), aber mit echten Kategorien aus contextual_data
    # =============================================================================
    print("\nZusatzplots: Verteilung Jahresenergie & Peak (nach unit_type) ...")

    # Jahresenergie pro Meter aus kumulierter Energie "Energi 1 Varmeenergi" innerhalb 2019:
    # annual_kwh_2019 = max(Energi1) - min(Energi1) pro Meter im Jahr 2019.
    # Warum: unabhängig von fehlenden Stunden in Effekt1; entspricht eher einem Zählerstand.
    # Voraussetzung: Energi 1 ist (nahezu) monoton und wird nicht zurückgesetzt (du meintest: nein/selten).

    # Wir laden Energi 1 zusätzlich (nur für 2019) und aggregieren direkt in einem groupby.
    print("  Lade Energi 1 Varmeenergi für Jahresenergie-Plot ...")

    # df_raw enthält nur 3 Spalten (Zeit, MeterID, Effekt 1). Für Energi 1 müssen wir erneut lesen –
    # aber wieder nur 3 Spalten und nur die Zeitspanne 2019 (Filter danach).
    # Performance-Hinweis: Das ist ein zusätzlicher Pass über die CSVs, aber immer noch relativ schlank.

    energi_chunks = []
    for i, f in enumerate(files, 1):
        sep = detect_csv_separator(f)
        d = pd.read_csv(
            f,
            sep=sep,
            usecols=[COL_DATETIME, COL_METER_ID, "Energi 1 Varmeenergi"],
            low_memory=False,
        )
        d.columns = ["datetime", "meter_id", "energi1_kwh"]
        d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", dayfirst=DAYFIRST)
        d["meter_id"] = d["meter_id"].astype(str)
        d["energi1_kwh"] = pd.to_numeric(d["energi1_kwh"], errors="coerce")
        energi_chunks.append(d)

    df_e = pd.concat(energi_chunks, ignore_index=True)
    df_e = df_e.dropna(subset=["datetime", "meter_id"])
    df_e_2019 = df_e[df_e["datetime"].dt.year == 2019].copy()

    energi_by_meter = df_e_2019.groupby("meter_id")["energi1_kwh"].agg(["min", "max"])
    annual_kwh = (energi_by_meter["max"] - energi_by_meter["min"]).rename("annual_kwh_2019")

    # Peak weiterhin aus Effekt 1 (Spitzenlast ist eine Leistungsgröße)
    peak_kw = df_hourly.max(axis=0, skipna=True).rename("peak_kw_2019")

    meter_stats = df_meta.set_index("meter_id").copy()
    meter_stats = meter_stats.join(annual_kwh, how="left")
    meter_stats = meter_stats.join(peak_kw, how="left")

    # Plausibilitäts-Checks
    print("  Meter mit annual_kwh_2019 NaN:", int(meter_stats["annual_kwh_2019"].isna().sum()))
    print("  Meter mit peak_kw_2019 NaN:", int(meter_stats["peak_kw_2019"].isna().sum()))
    # Negativwerte deuten auf Reset/Fehler hin
    n_neg = int((meter_stats["annual_kwh_2019"] < 0).sum())
    if n_neg:
        print(f"  Warnung: {n_neg} Meter mit negativer Jahresenergie (möglicher Reset/Fehler)")

    # Farbmap für unit_type
    unit_types_sorted = sorted(meter_stats["unit_type"].fillna("unbekannt").unique())
    palette = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#E91E63", "#00BCD4", "#607D8B"]
    type_color = {t: palette[i % len(palette)] for i, t in enumerate(unit_types_sorted)}

    # 3er-Panel wie im Fremd-Code
    figx, axes = plt.subplots(1, 3, figsize=(18, 6))
    figx.suptitle("Aalborg Smart Meter — Verteilung & Peak nach unit_type (2019, Effekt 1)", fontsize=14, fontweight="bold")

    # (1) Histogramm Jahresenergie [MWh]
    ax = axes[0]
    for ut in unit_types_sorted:
        sub = meter_stats.loc[meter_stats["unit_type"] == ut, "annual_kwh_2019"].dropna() / 1000
        if len(sub):
            ax.hist(sub, bins=30, color=type_color[ut], alpha=0.7, label=ut)
    ax.set_xlabel("Jahresenergie [MWh/Jahr]")
    ax.set_ylabel("Anzahl Meter")
    ax.set_title("Verteilung Jahresenergie")
    ax.legend(fontsize=8)

    # (2) Scatter: Jahresenergie vs Peak
    ax = axes[1]
    for ut in unit_types_sorted:
        grp = meter_stats[meter_stats["unit_type"] == ut].dropna(subset=["annual_kwh_2019", "peak_kw_2019"])
        if len(grp):
            ax.scatter(grp["annual_kwh_2019"] / 1000, grp["peak_kw_2019"],
                       label=ut, color=type_color[ut], s=10, alpha=0.6)
    ax.set_xlabel("Jahresenergie [MWh/Jahr]")
    ax.set_ylabel("Spitzenlast [kW]")
    ax.set_title("Energie vs. Spitzenlast")
    ax.legend(fontsize=8)

    # (3) Pie: Verteilung unit_type
    ax = axes[2]
    counts = meter_stats["unit_type"].fillna("unbekannt").value_counts().reindex(unit_types_sorted).fillna(0).astype(int)
    ax.pie(counts.values, labels=counts.index, colors=[type_color[t] for t in counts.index],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax.set_title("unit_type-Verteilung")

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # Original Step-1 Plots beibehalten
    # =============================================================================
    # PLOT 1 – Gesamtwärmebedarf
    print("\nPlot 1: Gesamtwärmebedarf ...")
    fig1, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df_hourly.index, total_heat.values, color="#1f77b4", linewidth=0.7)
    ax1.set_title("Gesamtwärmebedarf (Summe über alle Meter) – 2019", fontweight="bold")
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Wärmeleistung [kW]")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"NaN-Anteil (Stunden): {total_heat.isna().mean():.2%}\n"
        f"Max: {np.nanmax(total_heat.values):.1f} kW | Ø: {np.nanmean(total_heat.values):.1f} kW",
        xy=(0.01, 0.97), xycoords="axes fraction", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        fontsize=9,
    )
    fig1.tight_layout()
    plt.show()

    # PLOT 2 – Nach Haustyp + Pie (Stack)
    print("Plot 2: Nach Haustyp + Kuchendiagramm ...")
    types_sorted = sorted(heat_by_type.keys())
    color_map = {t: type_color.get(t, palette[i % len(palette)]) for i, t in enumerate(types_sorted)}

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 1]})

    bottom = np.zeros(len(df_hourly.index))
    for t in types_sorted:
        vals = heat_by_type[t].to_numpy()
        vals0 = np.nan_to_num(vals, nan=0.0)
        ax2a.fill_between(df_hourly.index, bottom, bottom + vals0, alpha=0.75, color=color_map[t], label=t)
        bottom += vals0

    ax2a.set_title("Gesamtwärmebedarf nach Haustyp – 2019", fontweight="bold")
    ax2a.set_xlabel("Datum")
    ax2a.set_ylabel("Wärmeleistung [kW]")
    ax2a.xaxis.set_major_locator(mdates.MonthLocator())
    ax2a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2a.legend(fontsize=9, loc="upper right")

    pie_labels, pie_vals, pie_cols = [], [], []
    for t in types_sorted:
        e = np.nansum(heat_by_type[t].values)
        if e > 0:
            pie_labels.append(t)
            pie_vals.append(e)
            pie_cols.append(color_map[t])

    ax2b.pie(pie_vals, labels=pie_labels, colors=pie_cols, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax2b.set_title("Anteile Haustypen\n(Jahresenergie, NaNs ignoriert)", fontweight="bold")

    fig2.tight_layout()
    plt.show()

    # PLOT 3 – Nach Baujahr-Klasse
    print("Plot 3: Nach Baujahr-Klasse ...")
    BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                     "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]
    b_palette = ["#7B1FA2", "#C62828", "#EF6C00", "#F9A825", "#558B2F", "#0277BD", "#00838F", "#9E9E9E"]
    b_color = {bk: b_palette[i % len(b_palette)] for i, bk in enumerate(BAUJAHR_ORDER)}

    fig3, ax3 = plt.subplots(figsize=(16, 6))
    bottom = np.zeros(len(df_hourly.index))
    for bk in BAUJAHR_ORDER:
        if bk not in heat_by_baujahr:
            continue
        vals = heat_by_baujahr[bk].to_numpy()
        vals0 = np.nan_to_num(vals, nan=0.0)
        ax3.fill_between(df_hourly.index, bottom, bottom + vals0, alpha=0.75, color=b_color[bk], label=bk)
        bottom += vals0

    ax3.set_title("Gesamtwärmebedarf nach Baujahr-Klasse – 2019", fontweight="bold")
    ax3.set_xlabel("Datum")
    ax3.set_ylabel("Wärmeleistung [kW]")
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax3.legend(fontsize=9, loc="upper right", title="Baujahr")

    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
