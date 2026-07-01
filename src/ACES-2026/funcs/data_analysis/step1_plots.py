#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot-Bibliothek für Step 1 (volle Aalborg-Daten), im Stil von funcs/plots.py.

Enthält je Plot eine eigene plot_*-Funktion mit show_plot-Parameter, genau wie
funcs/plots.py: einfache plt-Aufrufe (plot/hist/fill_between), Standard-
Matplotlib-Schrift (kein Calibri), Grid explizit je Achse via ax.grid(True),
plt.tight_layout()/constrained_layout und dieselben benannten Farben wie
plots.py (steelblue, tomato, gold, tab:green, darkorange, gray, black).

Die Daten werden aus dem Cache geladen, den step1_full_heat_demand.py am Ende
schreibt (step1_plot_cache.pkl). Direkt ausführen -> zeigt alle 4 Plots.
"""

import os

import numpy as np
import pandas as pd

# Interaktives GUI-Backend vor dem pyplot-Import setzen (eigenes Fenster je Plot).
import matplotlib
for _backend in ("Qt5Agg", "QtAgg", "TkAgg", "MacOSX"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Cache-Datei (von step1_full_heat_demand.py geschrieben, gleicher Ordner)
_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_HERE, "step1_plot_cache.pkl")


# --------------------------------------------------
# Anzeigenamen und Farben je Kategorie (benannte Farben wie plots.py)
# --------------------------------------------------
UNIT_TYPE_LABELS = {
    "apartment":           "Apartment",
    "single_family_house": "Single Family House",
    "terraced_house":      "Terraced House",
    "non_residential":     "Non-Residential",
    "unclear":             "Unclear",
    "unbekannt":           "Unknown",
}

# Feste Farbe je unit_type (nach Kategorie-NAME verankert) – dieselben benannten
# Farben, die auch in plots.py verwendet werden.
UNIT_TYPE_COLORS = {
    "single_family_house": "steelblue",
    "terraced_house":      "tomato",
    "apartment":           "tab:green",
    "non_residential":     "gold",
    "unclear":             "darkorange",
    "unbekannt":           "gray",
}
UNIT_TYPE_FALLBACK = ["tab:purple", "tab:brown", "tab:cyan", "tab:olive"]

# Reihenfolge und Anzeigenamen der Baujahr-Klassen
BAUJAHR_ORDER = ["vor 1919", "1919–1948", "1949–1968", "1969–1978",
                 "1979–1994", "1995–2009", "ab 2010", "Unbekannt"]
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
# Palette aus denselben benannten plots.py-Farben
BAUJAHR_PALETTE = ["steelblue", "tomato", "gold", "tab:green",
                   "darkorange", "gray", "tab:purple", "tab:brown"]
BAUJAHR_COLORS = {bk: BAUJAHR_PALETTE[i % len(BAUJAHR_PALETTE)]
                  for i, bk in enumerate(BAUJAHR_ORDER)}


def unit_label(ut):
    """Anzeigename je unit_type (ohne Unterstriche)."""
    return UNIT_TYPE_LABELS.get(str(ut), str(ut).replace("_", " ").title())


def type_colors(unit_types):
    """Farbe je unit_type: feste Zuordnung (UNIT_TYPE_COLORS), sonst Fallback."""
    cmap, k = {}, 0
    for t in unit_types:
        if t in UNIT_TYPE_COLORS:
            cmap[t] = UNIT_TYPE_COLORS[t]
        else:
            cmap[t] = UNIT_TYPE_FALLBACK[k % len(UNIT_TYPE_FALLBACK)]
            k += 1
    return cmap


# --------------------------------------------------
# Plot-Hilfsfunktionen (NaN-Achse, Tortenbeschriftung)
# --------------------------------------------------
def _nan_fraction_per_hour(df_hourly):
    """Anteil NaN-Meter je Stunde in Prozent (0% = alle vorhanden, 100% = alle NaN)."""
    return df_hourly.isna().mean(axis=1) * 100


def _add_nan_axis(ax_main, df_hourly, line_color="black"):
    """Zweite Y-Achse mit Anteil fehlender Meter je Stunde (gestrichelt, schwarz
    wie die Referenzlinien in plots.py). Gibt (Achse, Linien-Handle) zurück.
    """
    nan_frac = _nan_fraction_per_hour(df_hourly)
    ax_r = ax_main.twinx()
    ax_r.fill_between(df_hourly.index, nan_frac.values, alpha=0.10, color=line_color)
    line, = ax_r.plot(df_hourly.index, nan_frac.values, color=line_color,
                      linewidth=0.9, linestyle="--",
                      label="Share of meters without reading [%]")
    ax_r.set_ylabel("Share of meters without reading [%]", fontsize=9)
    ax_r.set_ylim(0, 100)
    return ax_r, line


def _place_outside_labels(ax, entries, side, radius, min_gap_frac=0.36):
    """Verteilt Außen-Labels einer Pie-Seite überlappungsfrei (vertikale Entzerrung)
    und verbindet jedes Label per Leitlinie mit seinem Segment.

    entries : Liste von (cy, cx, text) – cx/cy = Einheits-Mittelwinkel des Segments.
    side    : +1 = rechte Halbebene, -1 = linke Halbebene.
    """
    if not entries:
        return
    entries = sorted(entries, key=lambda e: e[0], reverse=True)  # oben -> unten
    min_gap = min_gap_frac * radius
    desired = [cy * radius for cy, _, _ in entries]
    ys = list(desired)
    for i in range(1, len(ys)):                      # Mindestabstand erzwingen
        if ys[i] > ys[i - 1] - min_gap:
            ys[i] = ys[i - 1] - min_gap
    shift = (sum(desired) - sum(ys)) / len(ys)       # Block zentrieren
    ys = [y + shift for y in ys]
    ylim = 1.45 * radius                             # in sichtbaren Bereich klemmen
    if ys[0] > ylim:
        ys = [y - (ys[0] - ylim) for y in ys]
    if ys[-1] < -ylim:
        ys = [y + (-ylim - ys[-1]) for y in ys]

    x_text = side * 1.22 * radius
    ha = "left" if side > 0 else "right"
    for (cy, cx, text), yt in zip(entries, ys):
        ax.annotate(
            text,
            xy=(cx * radius, cy * radius),           # Segment-Rand
            xytext=(x_text, yt),                     # entzerrte Position
            ha=ha, va="center", fontsize=9, linespacing=1.4,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8,
                            shrinkA=2, shrinkB=4,
                            connectionstyle="arc3,rad=0.0"),
        )


def _pie_with_outside_small_pct(ax, values, labels, colors,
                                threshold=8.0, startangle=90, radius=1.0):
    """Kreisdiagramm mit außenliegenden, überlappungsfreien Beschriftungen.

    - Große Segmente (>= threshold %): Prozentwert mittig im Segment, Name außen.
    - Kleine Segmente (< threshold %): Name + Prozentwert gemeinsam außen.
    """
    total = float(sum(values)) or 1.0
    wedges, _ = ax.pie(values, colors=colors, startangle=startangle,
                       radius=radius, textprops={"fontsize": 9})

    right, left = [], []
    for w, val, lab in zip(wedges, values, labels):
        pct = val / total * 100.0
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        cx, cy = np.cos(ang), np.sin(ang)
        if pct < threshold:
            text = f"{lab}\n{pct:.1f}%"
        else:
            ax.text(0.6 * radius * cx, 0.6 * radius * cy, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=9)
            text = f"{lab}"
        (right if cx >= 0 else left).append((cy, cx, text))

    _place_outside_labels(ax, right, side=+1, radius=radius)
    _place_outside_labels(ax, left,  side=-1, radius=radius)
    return wedges


# --------------------------------------------------
# Plot 1–3: Kennwerte je Haustyp (Histogramm, Scatter, Pie)
# --------------------------------------------------
def plot_stats_panel(meter_stats, show_plot=True):
    """Panel mit Histogramm (Jahresverbrauch), Scatter (Verbrauch vs. Spitzenlast)
    und Pie (Anzahl Meter je Haustyp)."""
    unit_types_sorted = sorted(meter_stats["unit_type"].fillna("unbekannt").unique().tolist())
    type_color = type_colors(unit_types_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(left=0.055, right=0.965, top=0.80, bottom=0.12, wspace=0.34)
    fig.suptitle("Full Heat Demand Data for 2019 per Type of House",
                 fontsize=14, fontweight="bold", y=0.96)

    # Plot 1: Histogramm Jahresverbrauch (kleinste Kategorie zuletzt/oben zeichnen)
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
    ax.grid(True)
    ax.legend(fontsize=8)

    # Plot 2: Scatter Jahresverbrauch vs. Spitzenlast
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
    ax.grid(True)
    ax.legend(fontsize=8)

    # Plot 3: Pie – Anzahl Meter je Haustyp
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
        radius=0.72,
    )
    ax.set_title("Share of Meters per Type of House", pad=18)

    if show_plot:
        plt.show()


# --------------------------------------------------
# Plot 4: Gesamtwärmeleistung + NaN-Anteil
# --------------------------------------------------
def plot_total_load(df_hourly, show_plot=True):
    """Zeitreihe der Gesamtwärmeleistung 2019 mit NaN-Hilfslinie."""
    total_heat = df_hourly.sum(axis=1, min_count=1)
    nan_frac   = _nan_fraction_per_hour(df_hourly)

    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    line_load, = ax.plot(df_hourly.index, total_heat.values, color="steelblue",
                         linewidth=0.8, label="Total heat load [kW]")
    ax.set_title("Total Heat Load in 2019", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Heat Load [kW]")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True)
    ax_r, line_nan = _add_nan_axis(ax, df_hourly)
    ax.annotate(
        f"NaN share total: {total_heat.isna().mean():.2%}\n"
        f"Avg. NaN meters per hour: {nan_frac.mean():.1f}%\n"
        f"Max: {np.nanmax(total_heat.values):.1f} kW  "
        f"| Avg.: {np.nanmean(total_heat.values):.1f} kW",
        xy=(0.5, 0.98), xycoords="axes fraction", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        fontsize=8,
    )
    ax.legend([line_load, line_nan],
              [line_load.get_label(), line_nan.get_label()],
              loc="upper center", bbox_to_anchor=(0.5, 0.78), fontsize=9)

    if show_plot:
        plt.show()


# --------------------------------------------------
# Plot 5: Gesamtlast nach Haustyp (Stack) + Pie Jahresenergie
# --------------------------------------------------
def plot_load_by_type(df_hourly, df_meta, show_plot=True):
    """Gestapelte Gesamtlast je Haustyp + Tortendiagramm des Jahresenergie-Anteils."""
    heat_by_type = {}
    for ut, mids in df_meta.groupby("unit_type")["meter_id"]:
        cols = [m for m in mids if m in df_hourly.columns]
        if cols:
            heat_by_type[ut] = df_hourly[cols].sum(axis=1, min_count=1)
    types_sorted = sorted(heat_by_type.keys())
    color_map    = type_colors(types_sorted)

    fig, (ax_stack, ax_pie) = plt.subplots(1, 2, figsize=(18, 6),
                                           gridspec_kw={"width_ratios": [3, 1.25]})
    fig.subplots_adjust(left=0.06, right=0.965, top=0.80, bottom=0.12, wspace=0.30)
    fig.suptitle("Full Heat Load per Type of House in 2019",
                 fontsize=13, fontweight="bold", y=0.98)

    # Stack-Flächen
    bottom = np.zeros(len(df_hourly.index))
    for t in types_sorted:
        vals = np.nan_to_num(heat_by_type[t].to_numpy(), nan=0.0)
        ax_stack.fill_between(df_hourly.index, bottom, bottom + vals,
                              alpha=0.85, color=color_map[t], label=unit_label(t))
        bottom += vals
    ax_r, line_nan = _add_nan_axis(ax_stack, df_hourly)
    ax_stack.set_xlabel("Month")
    ax_stack.set_ylabel("Heat Load [kW]")
    ax_stack.xaxis.set_major_locator(mdates.MonthLocator())
    ax_stack.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_stack.grid(True)
    handles, labels = ax_stack.get_legend_handles_labels()
    handles.append(line_nan)
    labels.append(line_nan.get_label())
    ax_stack.legend(handles, labels, fontsize=9, loc="upper right")

    # Pie: Anteil am Jahresenergiebedarf je Haustyp
    pie_labels, pie_vals, pie_cols = [], [], []
    for t in types_sorted:
        e = float(np.nansum(heat_by_type[t].values))
        if e > 0:
            pie_labels.append(unit_label(t))
            pie_vals.append(e)
            pie_cols.append(color_map[t])
    _pie_with_outside_small_pct(ax_pie, pie_vals, pie_labels, pie_cols, radius=0.85)
    ax_pie.set_title("Share of Annual Heat Demand", pad=26)

    if show_plot:
        plt.show()


# --------------------------------------------------
# Plot 6: Gesamtlast nach Baujahr-Klasse (Stack) + NaN-Anteil
# --------------------------------------------------
def plot_load_by_construction_year(df_hourly, df_meta, show_plot=True):
    """Gestapelte Gesamtlast je Baujahr-Klasse mit NaN-Hilfslinie."""
    heat_by_baujahr = {}
    for bk, mids in df_meta.groupby("baujahr_klasse")["meter_id"]:
        cols = [m for m in mids if m in df_hourly.columns]
        if cols:
            heat_by_baujahr[bk] = df_hourly[cols].sum(axis=1, min_count=1)

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    bottom = np.zeros(len(df_hourly.index))
    for bk in BAUJAHR_ORDER:
        if bk not in heat_by_baujahr:
            continue
        vals = np.nan_to_num(heat_by_baujahr[bk].to_numpy(), nan=0.0)
        ax.fill_between(df_hourly.index, bottom, bottom + vals,
                        alpha=0.85, color=BAUJAHR_COLORS[bk],
                        label=BAUJAHR_LABELS_EN.get(bk, bk))
        bottom += vals
    ax_r, line_nan = _add_nan_axis(ax, df_hourly)
    ax.set_title("Full Heat Load of 2019 per Construction Year", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Heat Load [kW]")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(line_nan)
    labels.append(line_nan.get_label())
    ax.legend(handles, labels, fontsize=8, loc="upper center", ncol=3,
              title="Construction Year", framealpha=0.9)

    if show_plot:
        plt.show()


# --------------------------------------------------
# Daten laden
# --------------------------------------------------
def load_cached_data():
    """Lädt die von step1 gespeicherten Plot-Daten aus dem Cache.

    Returns: (df_hourly, meter_stats, df_meta)
    """
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            f"Plot-Cache nicht gefunden: {CACHE_FILE}\n"
            "Bitte zuerst step1_full_heat_demand.py einmal vollständig laufen lassen "
            "(es schreibt den Cache am Ende automatisch)."
        )
    cache = pd.read_pickle(CACHE_FILE)
    return cache["df_hourly"], cache["meter_stats"], cache["df_meta"]


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    df_hourly, meter_stats, df_meta = load_cached_data()
    print(f"Geladen aus Cache: {df_hourly.shape[1]} Meter, {df_hourly.shape[0]} Stunden. "
          "Erzeuge Plots ...")

    plot_stats_panel(meter_stats)
    plot_total_load(df_hourly)
    plot_load_by_type(df_hourly, df_meta)
    plot_load_by_construction_year(df_hourly, df_meta)


if __name__ == "__main__":
    main()
