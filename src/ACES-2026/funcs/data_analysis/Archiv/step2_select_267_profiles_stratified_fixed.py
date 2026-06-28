#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 – Auswahl von 267 repräsentativen Lastprofilen (Option A: stratifiziert)

Ziel
----
Aus ~3000 Smart-Meter-Profilen werden 267 Profile ausgewählt, die die Aalborg-Realität
möglichst gut abbilden – passend zu einem Beispiel-Dorf mit 267 Haushalten.

Methodik (empfohlen, robust, gut begründbar)
--------------------------------------------
1) Exakt proportionale Stichprobe nach unit_type (Haustyp-Verteilung bleibt erhalten)
2) Innerhalb jedes unit_type: Stratifizierung nach Jahresverbrauch (Energi 1 Varmeenergi)
   über Quantil-Bins (hier: 10 Bins)
3) Zufällige Auswahl innerhalb jeder Schicht (reproduzierbar über Seed)

Umgang mit fehlenden Stunden
----------------------------
- Die ausgewählten 267 stündlichen Profile für 2019 werden auf 8760 Stunden gebracht.
- Fehlende Werte werden NUR für kurze Lücken interpoliert (max. 6h am Stück).
  Größere Lücken bleiben NaN.

Outputs
-------
- CSV long: [datetime, meter_id, effekt1_kw] für die 267 ausgewählten Meter (2019, stündlich)
- Meta: meter_id → unit_type, construction_year, annual_kwh_2019, peak_kw_2019, chosen, bin

Hinweis
-------
Dieses Skript liest die Smart-Meter-CSV-Dateien erneut ein (nur benötigte Spalten) und ist
unabhängig von deinem Step-1-Plot-Skript.
"""

from __future__ import annotations

import os
import glob
import csv

import numpy as np
import pandas as pd

# =============================================================================
# 0) KONFIGURATION – anpassen
# =============================================================================

DATA_DIR = r"/Users/nele/Documents/GitHub/ACES-2026/src/ACES-2026/Data/Aalborg_smart_meter_data"  # <-- ANPASSEN
CONTEXT_FILENAME = "contextual_data.csv"

# Smart-Meter Spalten
COL_DATETIME = "RoundedReadTime"
COL_METER_ID = "MeterID"
COL_EFFEKT1 = "Effekt 1"                  # kW
COL_ENERGI1 = "Energi 1 Varmeenergi"      # kWh (kumuliert)
COL_MAX_EFFEKT = "Maks.-effekt 1"         # kW

# Kontextdaten
CTX_METER_ID = "meter_id"
CTX_UNIT_TYPE = "unit_type"
CTX_CONSTRUCTION = "construction_year"

DAYFIRST = True

# Sampling
N_TARGET = 267
N_BINS = 10
RANDOM_SEED = 42

# NaN/Interpolation Policy
INTERP_MAX_GAP_HOURS = 6  # max. zusammenhaengende NaN-Stunden, die gefuellt werden

# Output
OUT_LONG = os.path.join(DATA_DIR, "selected_267_profiles_2019_long.csv")
OUT_META = os.path.join(DATA_DIR, "selected_267_profiles_meta.csv")


# =============================================================================
# 1) Hilfsfunktionen
# =============================================================================

def detect_csv_separator(path: str, default: str = ",") -> str:
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        return default


def list_meter_files(data_dir: str) -> list[str]:
    """Listet nur die echten Rohdaten-CSVs der Smart-Meter.

    Warum: Im DATA_DIR liegen inzwischen auch Output-Dateien (selected_*.csv),
    die nicht das Rohdaten-Schema haben.
    """
    files = glob.glob(os.path.join(data_dir, "*.csv"))

    # Kontext- und Output-Dateien ausschliessen
    exclude = {
        CONTEXT_FILENAME,
        os.path.basename(OUT_LONG),
        os.path.basename(OUT_META),
    }
    # Alles, was mit selected_ beginnt, ebenfalls ausschliessen
    files = [
        f for f in files
        if os.path.basename(f) not in exclude
        and not os.path.basename(f).startswith("selected_")
    ]

    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"Keine Rohdaten-Smart-Meter-CSV-Dateien in '{data_dir}' gefunden. "
            "(Hinweis: selected_*.csv und contextual_data.csv werden ignoriert.)"
        )
    return files


def parse_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(",", ".", regex=False)
    s2 = s2.str.replace(r"[^\d\.\-]", "", regex=True)
    s2 = s2.replace("", np.nan)
    return pd.to_numeric(s2, errors="coerce")


def load_context(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, CONTEXT_FILENAME)
    sep = detect_csv_separator(path)
    df_ctx = pd.read_csv(path, sep=sep, low_memory=False)

    missing = [c for c in [CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION] if c not in df_ctx.columns]
    if missing:
        raise KeyError("Kontext-Spalten fehlen: " + ", ".join(missing))

    df_ctx = df_ctx[[CTX_METER_ID, CTX_UNIT_TYPE, CTX_CONSTRUCTION]].copy()
    df_ctx.columns = ["meter_id", "unit_type", "construction_year"]
    df_ctx["meter_id"] = df_ctx["meter_id"].astype(str).str.strip()
    df_ctx["unit_type"] = df_ctx["unit_type"].astype(str)
    df_ctx["construction_year"] = pd.to_numeric(df_ctx["construction_year"], errors="coerce")
    return df_ctx


def annual_kwh_from_energi1_2019(df_2019: pd.DataFrame) -> pd.Series:
    e = df_2019.dropna(subset=["energi1_kwh"]).groupby("meter_id")["energi1_kwh"].agg(["min", "max"])
    annual = (e["max"] - e["min"]).rename("annual_kwh_2019")
    return annual


def peak_kw_from_max_effekt_2019(df_2019: pd.DataFrame) -> pd.Series:
    return df_2019.groupby("meter_id")["max_effekt_kw"].max().rename("peak_kw_2019")


def build_hourly_effect1(df_2019: pd.DataFrame) -> pd.DataFrame:
    df = df_2019[["datetime", "meter_id", "effekt1_kw"]].copy()
    df["datetime_h"] = df["datetime"].dt.floor("h")
    df_agg = df.groupby(["datetime_h", "meter_id"], as_index=False)["effekt1_kw"].mean()
    df_hourly = df_agg.pivot(index="datetime_h", columns="meter_id", values="effekt1_kw")
    full_index = pd.date_range("2019-01-01 00:00", "2019-12-31 23:00", freq="h")
    df_hourly = df_hourly.reindex(full_index)
    df_hourly.index.name = "datetime"
    return df_hourly


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Füllt nur kurze NaN-Lücken bis max_gap (in Stunden). Größere bleiben NaN."""
    s = series.copy()
    is_na = s.isna().to_numpy()
    if not is_na.any():
        return s

    # Zeitinterpolation, aber limit verhindert, dass lange Lücken gefüllt werden.
    s = s.interpolate(method="time", limit=max_gap, limit_direction="both")
    return s


def proportional_allocation(counts: pd.Series, n_target: int) -> pd.Series:
    """Exakt proportionale Allokation per Largest Remainder (Hamilton)."""
    weights = counts / counts.sum()
    raw = weights * n_target
    base = np.floor(raw).astype(int)
    remainder = raw - base

    missing = n_target - base.sum()
    if missing > 0:
        add_idx = remainder.sort_values(ascending=False).head(missing).index
        base.loc[add_idx] += 1
    elif missing < 0:
        sub_idx = remainder.sort_values(ascending=True).head(-missing).index
        base.loc[sub_idx] -= 1

    return base


def stratified_sample_within_type(df_type: pd.DataFrame, n_pick: int, n_bins: int, rng: np.random.Generator) -> pd.Index:
    """Wählt n_pick Meter aus df_type stratifiziert nach annual_kwh_2019 (Quantile)."""
    df_type = df_type.dropna(subset=["annual_kwh_2019"]).copy()
    if len(df_type) == 0 or n_pick <= 0:
        return pd.Index([])

    bins = min(n_bins, max(1, df_type["annual_kwh_2019"].nunique()))

    try:
        df_type["bin"] = pd.qcut(df_type["annual_kwh_2019"], q=bins, duplicates="drop")
    except ValueError:
        df_type["bin"] = "all"

    bin_counts = df_type["bin"].value_counts().sort_index()
    alloc = proportional_allocation(bin_counts, n_pick)

    picks = []
    for b, k in alloc.items():
        if k <= 0:
            continue
        pool = df_type[df_type["bin"] == b].index.to_numpy()
        if k >= len(pool):
            picks.extend(pool.tolist())
        else:
            picks.extend(rng.choice(pool, size=k, replace=False).tolist())

    return pd.Index(picks)


# =============================================================================
# 2) MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Step 2: Auswahl 267 repräsentative Profile (stratifiziert)")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_SEED)

    files = list_meter_files(DATA_DIR)
    print(f"CSV-Dateien (Smart Meter): {len(files)}")

    # Kontext
    df_ctx = load_context(DATA_DIR)

    # Smart-Meter Daten laden (nur benötigte Spalten) + Filter 2019
    chunks = []
    for i, f in enumerate(files, 1):
        sep = detect_csv_separator(f)
        print(f"  Lese {i:02d}/{len(files)}: {os.path.basename(f)} (sep='{sep}')")

        # --- Robust: Header prüfen (BOM/Whitespace/abweichende Dateien erkennen) ---
        expected = [COL_DATETIME, COL_METER_ID, COL_EFFEKT1, COL_ENERGI1, COL_MAX_EFFEKT]

        cols = pd.read_csv(f, sep=sep, nrows=0).columns.tolist()
        cols_norm = [c.replace("\ufeff", "").strip() for c in cols]
        missing = [c for c in expected if c not in cols_norm]

        if missing:
            raise ValueError(
                f"\nDatei hat nicht die erwarteten Spalten: {os.path.basename(f)}\n"
                f"Erkannt (raw): {cols}\n"
                f"Erkannt (norm): {cols_norm}\n"
                f"Fehlend: {missing}\n"
                f"Verwendeter sep='{sep}'\n"
                "=> Prüfe Separator/Dateityp oder passe COL_* an.\n"
            )

        d_full = pd.read_csv(f, sep=sep, low_memory=False)
        rename_map = {c: c.replace("\ufeff", "").strip() for c in d_full.columns}
        d_full = d_full.rename(columns=rename_map)

        d = d_full[expected].copy()
        d = d.rename(columns={
            COL_DATETIME: "datetime",
            COL_METER_ID: "meter_id",
            COL_EFFEKT1: "effekt1_kw",
            COL_ENERGI1: "energi1_kwh",
            COL_MAX_EFFEKT: "max_effekt_kw",
        })

        d["meter_id"] = d["meter_id"].astype(str).str.strip()
        d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", dayfirst=DAYFIRST)
        d["effekt1_kw"] = parse_numeric_series(d["effekt1_kw"])
        d["energi1_kwh"] = parse_numeric_series(d["energi1_kwh"])
        d["max_effekt_kw"] = parse_numeric_series(d["max_effekt_kw"])

        d = d.dropna(subset=["datetime", "meter_id"])
        d = d[d["datetime"].dt.year == 2019]
        chunks.append(d)

    df_2019 = pd.concat(chunks, ignore_index=True)

    print(f"\nZeilen 2019: {len(df_2019):,}")
    print(f"Meter 2019:  {df_2019['meter_id'].nunique():,}")

    # Kennwerte pro Meter
    annual = annual_kwh_from_energi1_2019(df_2019)
    peak = peak_kw_from_max_effekt_2019(df_2019)

    # Meta-Tabelle
    meta = (
        pd.DataFrame({"meter_id": df_2019["meter_id"].unique()})
        .merge(df_ctx, on="meter_id", how="left")
        .set_index("meter_id")
        .join(annual, how="left")
        .join(peak, how="left")
    )

    meta["unit_type"] = meta["unit_type"].fillna("unbekannt")

    print("\nJoin-Check: Missing unit_type:", int(meta["unit_type"].isna().sum()))
    print("Join-Check: Missing annual_kwh_2019:", int(meta["annual_kwh_2019"].isna().sum()))

    # Allokation exakt proportional nach unit_type
    type_counts = meta["unit_type"].value_counts()
    alloc_types = proportional_allocation(type_counts, N_TARGET)

    print("\nAllokation nach unit_type (exakt proportional):")
    print(alloc_types.to_string())
    print("Summe:", int(alloc_types.sum()))

    # Auswahl stratifiziert innerhalb jedes unit_type
    chosen = []
    meta2 = meta.copy()
    meta2["chosen"] = False
    meta2["bin"] = pd.NA

    for ut, n_pick in alloc_types.items():
        pool = meta2[meta2["unit_type"] == ut]
        picked_idx = stratified_sample_within_type(pool, int(n_pick), N_BINS, rng)
        meta2.loc[picked_idx, "chosen"] = True
        chosen.extend(picked_idx.tolist())

        sub = pool.dropna(subset=["annual_kwh_2019"]).copy()
        if len(sub) > 0:
            try:
                sub["bin"] = pd.qcut(
                    sub["annual_kwh_2019"],
                    q=min(N_BINS, sub["annual_kwh_2019"].nunique()),
                    duplicates="drop",
                )
                meta2.loc[sub.index, "bin"] = sub["bin"].astype(str)
            except Exception:
                pass

    chosen = pd.Index(chosen).unique()

    # Sicherheit: exakt N_TARGET
    if len(chosen) != N_TARGET:
        missing = N_TARGET - len(chosen)
        if missing > 0:
            rest = meta2[~meta2["chosen"]].dropna(subset=["annual_kwh_2019"]).index
            add = rng.choice(rest.to_numpy(), size=missing, replace=False)
            meta2.loc[add, "chosen"] = True
            chosen = pd.Index(list(chosen) + list(add))
        else:
            chosen = pd.Index(rng.choice(chosen.to_numpy(), size=N_TARGET, replace=False))
            meta2["chosen"] = False
            meta2.loc[chosen, "chosen"] = True

    print("\nAusgewählte Meter:", len(chosen))

    # Stündliche Effekt1-Profile bauen und auf Auswahl reduzieren
    df_hourly = build_hourly_effect1(df_2019)
    df_sel = df_hourly.reindex(columns=[m for m in chosen if m in df_hourly.columns])

    # Interpolation kurzer Lücken pro Meter (ohne Fragmentierung)
    print(f"\nInterpolation kurzer Lücken: max {INTERP_MAX_GAP_HOURS}h ...")
    df_sel = pd.DataFrame(
        {col: interpolate_short_gaps(df_sel[col], INTERP_MAX_GAP_HOURS) for col in df_sel.columns},
        index=df_sel.index,
    )

    # --- VALIDIERUNG (kurz) ---
    print("\nVALIDIERUNG: Sample vs. Original")
    sample_meta = meta2[meta2["chosen"]]

    print("\nunit_type-Anteile [%] (Original):")
    print((meta2["unit_type"].value_counts(normalize=True) * 100).round(1))

    print("\nunit_type-Anteile [%] (Sample):")
    print((sample_meta["unit_type"].value_counts(normalize=True) * 100).round(1))

    print("\nNaN-Anteil nach Interpolation (gesamt):", float(df_sel.isna().mean().mean()))

    # Output long (pandas >= 2.2 robust)
    out_long = (
        df_sel
        .reset_index()
        .melt(id_vars=["datetime"], var_name="meter_id", value_name="effekt1_kw")
    )

    out_long.to_csv(OUT_LONG, index=False)

    # Meta export
    meta_out = meta2.reset_index().rename(columns={"index": "meter_id"})
    meta_out.to_csv(OUT_META, index=False)

    print("\nGespeichert:")
    print("  Profile (long):", OUT_LONG)
    print("  Meta:", OUT_META)


if __name__ == "__main__":
    main()
