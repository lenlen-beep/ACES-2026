"""
=============================================================================
ACES Projekt 2026 – Schritt 1: Wärmelastdaten 2019 je Smart-Meter
=============================================================================
Dieses Skript:
  1. Liest alle Smart-Meter-CSV-Dateien ein (3021 Meter, 3 Jahre, stündlich)
  2. Filtert auf das Jahr 2019 und die relevanten Spalten
  3. Verknüpft die Meter-Daten mit den Hausinformationen (unit_type, construction)
  4. Erstellt einen vollständigen stündlichen Zeitindex für 2019 (8760 h)
  5. Interpoliert fehlende Stundenwerte (z.B. wenn ein Meter erst im Feb. startet)
  6. Erzeugt drei Plots:
       Plot 1 – Gesamtwärmebedarf aller Häuser 2019 (stündlich, kW)
       Plot 2 – Gesamtwärmebedarf nach Haustyp (farblich) + Kuchendiagramm
       Plot 3 – Gesamtwärmebedarf nach Baujahr-Klasse (farblich)
=============================================================================
ANPASSUNGEN NÖTIG (mit "# <-- ANPASSEN" markiert):
  - Pfad zu den Smart-Meter-CSV-Dateien
  - Pfad zur Kontextdaten-CSV
  - Spaltennamen (falls abweichend von Screenshot)
=============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0. KONFIGURATION – hier anpassen
# =============================================================================

# Gemeinsamer Ordner für ALLE Dateien (Smart-Meter-CSVs + contextual_data.csv)
DATA_DIR = r"/Users/nele/Documents/GitHub/ACES-2026/src/ACES-2026/Data/Aalborg_smart_meter_data"

# Exakter Dateiname der Kontextdaten-CSV (liegt im gleichen Ordner)
CONTEXT_FILENAME = "contextual_data.csv"

# Alle anderen CSVs im Ordner werden automatisch als Smart-Meter-Daten eingelesen
# (alles außer der Kontextdaten-Datei)

# Pfade werden automatisch zusammengesetzt:
CONTEXT_CSV     = os.path.join(DATA_DIR, CONTEXT_FILENAME)
SINGLE_CSV_FILE = None   # nicht nötig, da Ordner verwendet wird

# Spaltennamen in den Smart-Meter-Daten (aus Screenshot abgelesen)
COL_DATETIME  = "RoundedReadTime"   # Datum+Uhrzeit-Spalte  <-- ANPASSEN falls nötig
COL_METER_ID  = "MeterID"           # Meter-ID-Spalte        <-- ANPASSEN falls nötig
COL_EFFEKT1   = "Effekt 1"           # Wärmeleistung in kW    <-- ANPASSEN falls nötig

# Spaltennamen in der Kontextdaten-CSV (aus Screenshot abgelesen)
CTX_METER_ID    = "meter_id"        # <-- ANPASSEN falls nötig
CTX_UNIT_TYPE   = "unit_type"       # Haustyp (z.B. single_family, terraced_house)
CTX_CONSTRUCTION = "construction_year"   # Baujahr

# Trennzeichen der CSV-Dateien
CSV_SEPARATOR = ","                  # <-- ANPASSEN falls nötig (z.B. ";" für deutsche CSVs)

# Ausgabeordner für Plots
OUTPUT_DIR = "./plots_step1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. SMART-METER-DATEN EINLESEN
# =============================================================================
print("Schritt 1: Smart-Meter-Daten einlesen ...")

# Alle CSV-Dateien im Ordner finden, Kontextdaten-Datei ausschließen
csv_files = [
    f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if os.path.basename(f) != CONTEXT_FILENAME
]

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"Keine Smart-Meter-CSV-Dateien in '{DATA_DIR}' gefunden "
        f"(außer '{CONTEXT_FILENAME}'). Bitte DATA_DIR anpassen."
    )

print(f"  Gefundene Smart-Meter-Dateien: {len(csv_files)}")
for f in csv_files:
    print(f"    - {os.path.basename(f)}")

# In der Einlese-Schleife (Schritt 1) – nur relevante Spalten laden:
chunks = []
for i, f in enumerate(csv_files):
    print(f"  Lese Datei {i+1}/{len(csv_files)}: {os.path.basename(f)} ...")
    chunks.append(
        pd.read_csv(
            f,
            sep=CSV_SEPARATOR,
            usecols=[COL_DATETIME, COL_METER_ID, COL_EFFEKT1],  # <-- nur 3 Spalten!
            low_memory=False
        )
    )

df_raw = pd.concat(chunks, ignore_index=True)
print(f"  Rohdaten geladen: {len(df_raw):,} Zeilen, {df_raw.shape[1]} Spalten")
print(f"  Spalten: {list(df_raw.columns)}")

# =============================================================================
# 2. RELEVANTE SPALTEN AUSWÄHLEN & DATENTYPEN BEREINIGEN
# =============================================================================
print("\nSchritt 2: Spalten filtern und Datentypen bereinigen ...")

# Nur benötigte Spalten behalten
df = df_raw[[COL_DATETIME, COL_METER_ID, COL_EFFEKT1]].copy()
df.columns = ["datetime", "meter_id", "effekt1_kw"]

# Datetime parsen (flexibel – pandas erkennt die meisten Formate automatisch)
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)

# Effekt1 in numerisch umwandeln (Einheit kW laut Screenshot)
# Falls die Spalte Einheiten enthält (z.B. "2.9 kW"), werden sie entfernt
df["effekt1_kw"] = (
    df["effekt1_kw"]
    .astype(str)
    .str.replace(r"[^\d\.\-]", "", regex=True)
    .replace("", np.nan)
    .astype(float)
)

# Ungültige Zeilen entfernen
n_before = len(df)
df = df.dropna(subset=["datetime", "meter_id"])
print(f"  {n_before - len(df):,} Zeilen mit ungültigem Datum/Meter-ID entfernt")

# =============================================================================
# 3. AUF JAHR 2019 FILTERN
# =============================================================================
print("\nSchritt 3: Auf Jahr 2019 filtern ...")

df_2019 = df[df["datetime"].dt.year == 2019].copy()
print(f"  Zeilen für 2019: {len(df_2019):,}")
print(f"  Unique Meter-IDs in 2019: {df_2019['meter_id'].nunique():,}")

# --- Check: Stunden-Abdeckung 2019 je Meter ---------------------------------
df_2019["datetime_h"] = df_2019["datetime"].dt.floor("h")

# Anzahl vorhandener Stunden pro Meter (unique Stundenstempel)
hours_per_meter = df_2019.groupby("meter_id")["datetime_h"].nunique()

expected_hours = 8760  # 2019 ist kein Schaltjahr
missing_hours = expected_hours - hours_per_meter

print("\nCHECK: Stunden-Abdeckung 2019 je Meter")
print(f"  Meter insgesamt (mit Daten in 2019): {hours_per_meter.size}")
print(f"  Meter mit vollständigen 8760h:       {(missing_hours==0).sum()}")
print(f"  Meter mit fehlenden Stunden:         {(missing_hours>0).sum()}")
print(f"  Max fehlende Stunden (worst case):   {missing_hours.max()}")
print("  Top 10 Meter mit den meisten fehlenden Stunden:")
print(missing_hours.sort_values(ascending=False).head(10))

counts_per_hour = df_2019.groupby(["meter_id", "datetime_h"]).size()
print("\nCHECK: Mehrfachmessungen pro Stunde?")
print(counts_per_hour.describe())
print("  Anteil Stunden mit >1 Messung:", (counts_per_hour > 1).mean())

# =============================================================================
# 4. KONTEXTDATEN EINLESEN UND VERKNÜPFEN
# =============================================================================
print("\nSchritt 4: Kontextdaten einlesen und verknüpfen ...")

df_ctx = pd.read_csv(CONTEXT_CSV, sep=CSV_SEPARATOR, low_memory=False)
print(f"  Kontextdaten: {len(df_ctx):,} Einträge, Spalten: {list(df_ctx.columns)}")

# Nur benötigte Spalten
df_ctx = df_ctx[[CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION]].copy()
df_ctx.columns = ["meter_id", "unit_type", "construction_year"]
df_ctx["meter_id"] = df_ctx["meter_id"].astype(str)
df_ctx["construction_year"] = pd.to_numeric(df_ctx["construction_year"], errors="coerce")

df_2019["meter_id"] = df_2019["meter_id"].astype(str)
df_2019 = df_2019.merge(df_ctx, on="meter_id", how="left")

n_no_ctx = df_2019["unit_type"].isna().sum()
if n_no_ctx > 0:
    print(f"  Warnung: {n_no_ctx:,} Zeilen ohne Kontextdaten (unit_type = NaN)")
    df_2019["unit_type"] = df_2019["unit_type"].fillna("unbekannt")
    df_2019["construction_year"] = df_2019["construction_year"].fillna(-1)

print(f"  Haustypen: {df_2019['unit_type'].unique()}")

# =============================================================================
# 5. VOLLSTÄNDIGEN STÜNDLICHEN ZEITINDEX ERSTELLEN & FEHLENDE WERTE INTERPOLIEREN
# =============================================================================
print("\nSchritt 5: Vollständigen Zeitindex erstellen und fehlende Werte interpolieren ...")

# Vollständiger stündlicher Index für 2019 (8760 Stunden)
full_index = pd.date_range(start="2019-01-01 00:00", end="2019-12-31 23:00", freq="h")
print(f"  Vollständiger Index: {len(full_index)} Stunden")

# Datetime auf Stunde runden (statt resample in der Schleife)
df_2019["datetime_h"] = df_2019["datetime"].dt.floor("h")

# Schritt 5a: Auf Stundenbasis aggregieren – ALLE Meter auf einmal (vektorisiert)
print("  Aggregiere auf Stundenbasis (alle Meter gleichzeitig) ...")
df_agg = (
    df_2019
    .groupby(["meter_id", "datetime_h"])["effekt1_kw"]
    .mean()
    .reset_index()
)

# Schritt 5b: Pivot-Tabelle: Zeilen = Stunden, Spalten = Meter-IDs
print("  Erstelle Pivot-Tabelle ...")
df_hourly = df_agg.pivot_table(
    index="datetime_h",
    columns="meter_id",
    values="effekt1_kw",
    aggfunc="mean"
)

# Schritt 5c: Auf vollständigen Index reindexieren (fehlende Stunden = NaN)
print("  Reindexiere auf vollständigen Jahresindex ...")
df_hourly = df_hourly.reindex(full_index)
df_hourly.index.name = "datetime"

# Schritt 5d: Interpolation spaltenweise (alle Meter gleichzeitig)
print("  Interpoliere fehlende Werte ...")
df_hourly = df_hourly.interpolate(method="linear", limit_direction="both", axis=0)
df_hourly = df_hourly.fillna(0.0)

print(f"  Ergebnis-DataFrame: {df_hourly.shape[0]} Stunden x {df_hourly.shape[1]} Meter")

# Metadaten-DataFrame aus df_2019 ableiten
all_meters = df_hourly.columns.tolist()
df_meta = (
    df_2019[["meter_id", "unit_type", "construction_year"]]
    .drop_duplicates(subset="meter_id")
    .reset_index(drop=True)
)

# =============================================================================
# 6. AGGREGATION FÜR PLOTS
# =============================================================================
print("\nSchritt 6: Aggregation für Plots ...")

# Gesamtwärmebedarf (Summe aller Meter, stündlich)
total_heat = df_hourly.sum(axis=1)

# Wärmebedarf je Haustyp
unit_types = df_meta["unit_type"].unique()
heat_by_type = {}
for ut in unit_types:
    mids = df_meta[df_meta["unit_type"] == ut]["meter_id"].values
    mids_in_df = [m for m in mids if m in df_hourly.columns]
    heat_by_type[ut] = df_hourly[mids_in_df].sum(axis=1)

# Baujahr-Klassen definieren
def baujahr_klasse(year):
    if year < 0:
        return "Unbekannt"
    elif year < 1919:
        return "vor 1919"
    elif year < 1949:
        return "1919–1948"
    elif year < 1969:
        return "1949–1968"
    elif year < 1979:
        return "1969–1978"
    elif year < 1995:
        return "1979–1994"
    elif year < 2010:
        return "1995–2009"
    else:
        return "ab 2010"

BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                 "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]

df_meta["baujahr_klasse"] = df_meta["construction_year"].apply(baujahr_klasse)

heat_by_baujahr = {}
for bk in BAUJAHR_ORDER:
    mids = df_meta[df_meta["baujahr_klasse"] == bk]["meter_id"].values
    mids_in_df = [m for m in mids if m in df_hourly.columns]
    if mids_in_df:
        heat_by_baujahr[bk] = df_hourly[mids_in_df].sum(axis=1)

# =============================================================================
# 7. PLOT 1 – Gesamtwärmebedarf aller Häuser 2019
# =============================================================================
print("\nSchritt 7: Plot 1 – Gesamtwärmebedarf aller Häuser ...")

fig1, ax1 = plt.subplots(figsize=(16, 5))
ax1.fill_between(full_index, total_heat.values, alpha=0.6, color="#1f77b4", label="Gesamtwärmebedarf")
ax1.plot(full_index, total_heat.values, color="#1f77b4", linewidth=0.5)

ax1.set_title("Gesamtwärmebedarf aller Smart-Meter-Häuser – Jahr 2019", fontsize=14, fontweight="bold")
ax1.set_xlabel("Datum")
ax1.set_ylabel("Wärmeleistung [kW]")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Statistik-Annotation
ax1.annotate(
    f"Jahressumme: {total_heat.sum()/1000:.1f} MWh\n"
    f"Max: {total_heat.max():.1f} kW  |  Ø: {total_heat.mean():.1f} kW",
    xy=(0.01, 0.97), xycoords="axes fraction",
    va="top", ha="left", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "plot1_gesamtwaermebedarf.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Gespeichert: {OUTPUT_DIR}/plot1_gesamtwaermebedarf.png")

# =============================================================================
# 8. PLOT 2 – Gesamtwärmebedarf nach Haustyp + Kuchendiagramm
# =============================================================================
print("\nSchritt 8: Plot 2 – Wärmebedarf nach Haustyp + Kuchendiagramm ...")

# Farbpalette für Haustypen
TYPE_COLORS = {
    "single_family":  "#2196F3",
    "terraced_house": "#FF9800",
    "terraced_hou":   "#FF9800",   # abgekürzte Variante aus Screenshot
    "apartment":      "#4CAF50",
    "multi_family":   "#9C27B0",
    "unbekannt":      "#9E9E9E",
}
# Fallback-Farben für unbekannte Typen
FALLBACK_COLORS = ["#E91E63", "#00BCD4", "#8BC34A", "#FF5722", "#607D8B"]
fb_idx = 0
for ut in unit_types:
    if ut not in TYPE_COLORS:
        TYPE_COLORS[ut] = FALLBACK_COLORS[fb_idx % len(FALLBACK_COLORS)]
        fb_idx += 1

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(18, 6),
                                   gridspec_kw={"width_ratios": [3, 1]})

# Linkes Panel: gestapeltes Flächendiagramm nach Haustyp
bottom = np.zeros(len(full_index))
for ut in unit_types:
    if ut in heat_by_type:
        vals = heat_by_type[ut].values
        ax2a.fill_between(full_index, bottom, bottom + vals,
                          alpha=0.75, color=TYPE_COLORS.get(ut, "#999"),
                          label=ut)
        bottom += vals

ax2a.set_title("Gesamtwärmebedarf nach Haustyp – Jahr 2019", fontsize=13, fontweight="bold")
ax2a.set_xlabel("Datum")
ax2a.set_ylabel("Wärmeleistung [kW]")
ax2a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2a.xaxis.set_major_locator(mdates.MonthLocator())
ax2a.grid(True, alpha=0.3)
ax2a.legend(loc="upper right", fontsize=9)

# Rechtes Panel: Kuchendiagramm (Jahresenergie je Haustyp)
pie_labels = []
pie_values = []
pie_colors = []
for ut in unit_types:
    if ut in heat_by_type:
        total_ut = heat_by_type[ut].sum()
        if total_ut > 0:
            pie_labels.append(ut)
            pie_values.append(total_ut)
            pie_colors.append(TYPE_COLORS.get(ut, "#999"))

wedges, texts, autotexts = ax2b.pie(
    pie_values,
    labels=pie_labels,
    colors=pie_colors,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(9)
ax2b.set_title("Anteil Haustypen\n(Jahresenergie)", fontsize=11, fontweight="bold")

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "plot2_waermebedarf_haustyp.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Gespeichert: {OUTPUT_DIR}/plot2_waermebedarf_haustyp.png")

# =============================================================================
# 9. PLOT 3 – Gesamtwärmebedarf nach Baujahr-Klasse
# =============================================================================
print("\nSchritt 9: Plot 3 – Wärmebedarf nach Baujahr-Klasse ...")

# Farbpalette für Baujahr-Klassen (von alt = warm zu neu = kalt)
BAUJAHR_COLORS = {
    "vor 1919":   "#7B1FA2",
    "1919–1948":  "#C62828",
    "1949–1968":  "#EF6C00",
    "1969–1978":  "#F9A825",
    "1979–1994":  "#558B2F",
    "1995–2009":  "#0277BD",
    "ab 2010":    "#00838F",
    "Unbekannt":  "#9E9E9E",
}

fig3, ax3 = plt.subplots(figsize=(16, 6))

bottom = np.zeros(len(full_index))
for bk in BAUJAHR_ORDER:
    if bk in heat_by_baujahr:
        vals = heat_by_baujahr[bk].values
        ax3.fill_between(full_index, bottom, bottom + vals,
                         alpha=0.75, color=BAUJAHR_COLORS.get(bk, "#999"),
                         label=bk)
        bottom += vals

ax3.set_title("Gesamtwärmebedarf nach Baujahr-Klasse – Jahr 2019", fontsize=13, fontweight="bold")
ax3.set_xlabel("Datum")
ax3.set_ylabel("Wärmeleistung [kW]")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax3.xaxis.set_major_locator(mdates.MonthLocator())
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right", fontsize=9, title="Baujahr-Klasse")

# Jahresenergie je Klasse als Annotation
energy_lines = []
for bk in BAUJAHR_ORDER:
    if bk in heat_by_baujahr:
        e = heat_by_baujahr[bk].sum() / 1000
        energy_lines.append(f"{bk}: {e:.0f} MWh")

ax3.annotate(
    "\n".join(energy_lines),
    xy=(0.01, 0.97), xycoords="axes fraction",
    va="top", ha="left", fontsize=7.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85)
)

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "plot3_waermebedarf_baujahr.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Gespeichert: {OUTPUT_DIR}/plot3_waermebedarf_baujahr.png")

# =============================================================================
# 10. ZUSAMMENFASSUNG
# =============================================================================
print("\n" + "=" * 60)
print("ZUSAMMENFASSUNG")
print("=" * 60)
print(f"  Meter gesamt (2019):       {len(all_meters):>8,}")
print(f"  Stunden im Index:          {len(full_index):>8,}")
print(f"  Gesamtenergie 2019:        {total_heat.sum()/1000:>8.1f} MWh")
print(f"  Spitzenlast:               {total_heat.max():>8.1f} kW")
print(f"  Mittlere Leistung:         {total_heat.mean():>8.1f} kW")
print(f"\n  Wärmebedarf je Haustyp (Jahresenergie):")
for ut in unit_types:
    if ut in heat_by_type:
        e = heat_by_type[ut].sum() / 1000
        n = len(df_meta[df_meta["unit_type"] == ut])
        print(f"    {ut:<25} {e:>8.1f} MWh  ({n} Meter)")
print(f"\n  Plots gespeichert in: {os.path.abspath(OUTPUT_DIR)}/")
print("=" * 60)

# Optional: Stündliche Daten als CSV exportieren (für spätere Schritte)
EXPORT_CSV = os.path.join(OUTPUT_DIR, "hourly_heat_2019_all_meters.csv")
df_hourly.to_csv(EXPORT_CSV)
print(f"\n  Stündliche Daten exportiert: {EXPORT_CSV}")
print("  (Diese Datei wird in späteren Schritten weiterverwendet.)")
