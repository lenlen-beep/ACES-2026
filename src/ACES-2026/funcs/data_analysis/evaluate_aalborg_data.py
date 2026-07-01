import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "src/ACES-2026/Data/Aalborg_smart_meter_data/"
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
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
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
    cat_colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    color_map  = dict(zip(cat_order, cat_colors))

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        13,
        "axes.titlesize":   14,
        "axes.labelsize":   13,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.color":       "#DDDDDD",
        "grid.linewidth":   0.7,
        "legend.frameon":   False,
    })

    short_labels = [
        "EFH",
        "Kl. MFH",
        "Gr. MFH",
        "Gewerbe",
        "Industrie",
    ]
    short_map = dict(zip(cat_order, short_labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Aalborg Fernwärme — Gebäudekategorisierung (n = 3.127)",
                 fontsize=16, fontweight="bold", y=1.02)

    # ── 1. Balkendiagramm Anzahl je Kategorie ──────────────────────────────
    ax = axes[0]
    counts = (summary["Kategorie"]
              .value_counts()
              .reindex(cat_order)
              .fillna(0)
              .astype(int))
    bars = ax.barh([short_map[c] for c in cat_order],
                   [counts[c] for c in cat_order],
                   color=cat_colors, edgecolor="white", height=0.6)
    for bar, n in zip(bars, [counts[c] for c in cat_order]):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                f"{n:,}", va="center", fontsize=12)
    ax.set_xlabel("Anzahl Gebäude")
    ax.set_title("Anzahl je Kategorie")
    ax.set_xlim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0)

    # ── 2. Boxplot Jahresverbrauch je Kategorie ────────────────────────────
    ax = axes[1]
    data_by_cat = [
        (summary[summary["Kategorie"] == cat]["Jahresverbrauch_kWh"] / 1000).dropna().values
        for cat in cat_order
    ]
    bp = ax.boxplot(data_by_cat,
                    vert=True, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], cat_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.set_yscale("log")
    ax.set_xticklabels([short_map[c] for c in cat_order], rotation=20, ha="right")
    ax.set_ylabel("Jahresverbrauch [MWh/a]")
    ax.set_title("Verbrauchsverteilung je Kategorie")

    # ── 3. Scatter: Verbrauch vs. Spitzenlast ─────────────────────────────
    ax = axes[2]
    for cat, color in zip(cat_order, cat_colors):
        grp = summary[summary["Kategorie"] == cat].dropna(subset=["MaxLeistung_kW"])
        if len(grp):
            ax.scatter(grp["Jahresverbrauch_kWh"] / 1000,
                       grp["MaxLeistung_kW"],
                       label=short_map[cat], color=color,
                       s=18, alpha=0.65, linewidths=0)
    ax.set_xlabel("Jahresverbrauch [MWh/a]")
    ax.set_ylabel("Maximalleistung [kW]")
    ax.set_title("Verbrauch vs. Spitzenlast")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11, markerscale=1.8)

    plt.tight_layout()
    out = "src/ACES-2026/Data/aalborg_kategorisierung.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    plt.show()


if __name__ == "__main__":
    df      = load_all_files()
    summary = build_customer_summary(df)
    print_summary(summary)
    plot_results(summary)
