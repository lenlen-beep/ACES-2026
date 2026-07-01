#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zeigt die 6 Step-2-Plots erneut an – OHNE die volle Auswahl-Pipeline.

Statt alle Rohdaten neu einzulesen und die Stichprobe neu zu ziehen, werden nur
die bereits gespeicherten Ergebnis-CSVs geladen (selected_267_profiles_*) und die
identischen Plot-Funktionen aus step2_267_profiles_stratified.py aufgerufen.

Praktisch, wenn man eine Plot-Figur versehentlich weggeklickt hat: einfach dieses
Skript laufen lassen, dann öffnen sich alle Plot-Fenster wieder.
"""

import os
import importlib.util

import pandas as pd

# --------------------------------------------------
# Hauptskript als Modul laden (Backend, Plot-Funktionen, Pfade)
# --------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.join(_HERE, "step2_267_profiles_stratified.py")
_spec = importlib.util.spec_from_file_location("step2_main", _MAIN)
step2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step2)


# --------------------------------------------------
# Ergebnis-CSVs laden
# --------------------------------------------------
def load_selected_profiles():
    """Lädt Meta + stündliche Profile der 267 ausgewählten Meter aus den CSVs.

    Returns: (meta, df_hourly)
    """
    # --- Meta der ausgewählten Meter laden ---
    meta = pd.read_csv(step2.OUT_META)
    meta["meter_id"] = meta["meter_id"].astype(str)
    meta = meta.set_index("meter_id")

    # baujahr_klasse rekonstruieren, falls in der CSV nicht vorhanden
    if "baujahr_klasse" not in meta.columns:
        meta["baujahr_klasse"] = meta["construction_year"].apply(step2.baujahr_klasse)

    # --- Stündliche Profile aus der Long-CSV rekonstruieren ---
    long = pd.read_csv(step2.OUT_LONG, parse_dates=["datetime"])
    long["meter_id"] = long["meter_id"].astype(str)
    df_hourly = long.pivot(index="datetime", columns="meter_id", values="effekt1_kw")

    return meta, df_hourly


# --------------------------------------------------
# Plots erneut anzeigen
# --------------------------------------------------
def show_all_plots(meta, df_hourly, show_plot=True):
    """Erzeugt alle 6 Step-2-Plots mit den identischen Funktionen des Hauptskripts."""
    n = len(meta)
    step2.plot_stats_panel(meta, n)
    step2.plot_series_panels(df_hourly, meta, n)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    meta, df_hourly = load_selected_profiles()
    print(f"Geladen: {len(meta)} Meter, {df_hourly.shape[0]} Stunden. Erzeuge Plots ...")
    show_all_plots(meta, df_hourly)


if __name__ == "__main__":
    main()
