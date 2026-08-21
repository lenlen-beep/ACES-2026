#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Einheitlicher Plot-Stil für die Data-Processing-Skripte (Wärmelast Step 1/2,
Temperaturvergleich Aalborg/Jerrishoe). Einmal importieren (setzt beim Import
Schriftart/-größe global über rcParams); Farbpalette, PLOTS_DIR und _ppt_style
werden explizit in den jeweiligen Plot-Funktionen verwendet, damit alle
Abbildungen aus dem Bericht optisch identisch sind.
"""

import os

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

# Farbpalette Energiesystem
COLOR_WP       = "#00395B"   # EUF-Blau      — Wärmepumpe
COLOR_GAS      = "#C17A2F"   # Amber         — Gaskessel (fossil)
COLOR_SPEICHER = "#769D7B"   # Mint          — Kurzzeitspeicher
COLOR_SAISONAL = "#2F6B4F"   # Dunkelgrün    — Saisonalspeicher
COLOR_LAST     = "#1A1A1A"   # Fast-Schwarz  — Wärmebedarf (Referenz)
COLOR_VERLUST  = "#A0463A"   # Gedämpftes Rot — Netzverluste
COLOR_PV       = "#C8A84B"   # Gold          — PV / Sonne

# Zyklische Palette für Kategorien ohne feste Systemzuordnung (Haustypen,
# Baujahr-Klassen, Jahreszeiten, Stadtvergleich) – wird der Reihe nach vergeben.
PALETTE = [COLOR_WP, COLOR_GAS, COLOR_SPEICHER, COLOR_SAISONAL,
           COLOR_LAST, COLOR_VERLUST, COLOR_PV]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ACES_DIR = os.path.dirname(os.path.dirname(_HERE))  # .../src/ACES-2026
PLOTS_DIR = os.path.join(_ACES_DIR, "plots")

LABEL_FONTSIZE = 20
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 20
TITLE_FONTSIZE = 20


def cycle_colors(n):
    """Gibt n Farben aus PALETTE zurück (zyklisch, falls n > len(PALETTE))."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _ppt_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, color="#CCCCCC")
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.margins(x=0)


def _save(fig, filename):
    """Speichert die Figure als PNG in PLOTS_DIR."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {path}")
