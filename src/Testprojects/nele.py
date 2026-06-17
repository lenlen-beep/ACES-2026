#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:46:49 2026

@author: nele
"""

import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

"""
Optimierung eines Fernwärmesystems mit Wärmepumpe und Wärmespeicher.

Das Modell minimiert die Stromkosten einer Wärmepumpe unter Einhaltung der
Wärmeversorgung. Ein Speicher ermöglicht eine zeitliche Verschiebung.

Methodik:
- Daten: Zeitreihe der Wärmeleistung (Excel)
- Optimierung: Pyomo (lineares Optimierungsmodell)
- Ziel: Minimierung der Stromkosten
- Nebenbedingungen: Wärmebilanz + Speicherdynamik
"""

load_file  = r"/Users/nele/Library/Mobile Documents/com~apple~CloudDocs/02_Studium_Flensburg/02_SoSe 26/ACES/Aces 3c/Semesterproject/Codes and Data/district_heating_data_Flensburg_2017_SmallScale_Edited.xlsx"

# pd.readexcel(file,
#               skiprows: Zeilen überspringen (Anzahl),
#               header: Zeile mit Überschrift,
#               usecols:=[] Bestimmte Spaltenauswahl,
#               names=[]: Spalten direkt benennen,
#               parse_dates=[]: direkt in Datum umwandeln,
#               index_col: Index setzen,
#               na_values=['','']: Behandlung fehlender Werte,
#               nrows: Wertebereich laden (mit skiprows),
#   )

df_load = pd.read_excel(load_file, skiprows=1, header=0)

# Überschreiben Spaltennamen
df_load.columns = ['Datum', 'Wärmeleistung in MW']

df_load['Datum'] = pd.to_datetime(df_load['Datum'])

# Zeitindex
T = range(len(df_load))

# Nachfrage in einzelnen df
demand = df_load['Wärmeleistung in MW'].values 

#-----------------------------------------------------------------------------
# Strompreis aus Großhandelspreise-Excel laden (SMARD, DE-LU 2024)
#-----------------------------------------------------------------------------

price_file = r"/Users/nele/Library/Mobile Documents/com~apple~CloudDocs/02_Studium_Flensburg/02_SoSe 26/ACES/Aces 3c/Semesterproject/Codes and Data/Gro_handelspreise_202401010000_202501010000_Stunde.xlsx"

# skiprows=9: Die ersten 9 Zeilen sind Metadaten (Quelle, Zeitraum etc.)
# header=0:   Zeile 10 (nach dem Überspringen) ist die Spaltenüberschrift
df_price = pd.read_excel(price_file, skiprows=9, header=0)

# Nur die Deutschland/Luxemburg-Spalte verwenden
# pd.to_numeric(..., errors='coerce') wandelt '-' oder leere Felder in NaN um
df_price['price_eur_mwh'] = pd.to_numeric(
    df_price['Deutschland/Luxemburg [\u20ac/MWh]'], errors='coerce'
)

# Fehlende Werte (NaN) mit dem Mittelwert auffüllen (Fallback)
df_price['price_eur_mwh'] = df_price['price_eur_mwh'].fillna(
    df_price['price_eur_mwh'].mean()
)

# Auf 8.760 Stunden kürzen (Flensburg-Datensatz = 2017, kein Schaltjahr)
# Die Preise stammen aus 2024 (8.784 h wegen Schaltjahr) -> erste 8.760 nehmen
price_values = df_price['price_eur_mwh'].values[:len(T)]

# Als Dictionary für Pyomo: {Zeitschritt t: Preis in €/MWh}
price = {t: price_values[t] for t in T}

print(f"Strompreise geladen: {len(price)} Zeitschritte")
print(f"  Min: {min(price.values()):.2f} €/MWh")
print(f"  Max: {max(price.values()):.2f} €/MWh")
print(f"  Mittelwert: {sum(price.values())/len(price):.2f} €/MWh")

#-----------------------------------------------------------------------------
# Parameter Wärmepumpe
#-----------------------------------------------------------------------------

# --- Wärmepumpe: Technisch ---
COP = 3.5               # Coefficient of Performance
Q_wp_max = 5           # Max. thermische Leistung der WP [MW_th]

T_supply = 80           # Vorlauftemperatur Fernwärmenetz [°C] (Annahme)
T_return = 50           # Rücklauftemperatur Fernwärmenetz [°C] (Annahme)

# --- Wärmepumpe: Wirtschaftlich ---
lifetime = 20           # Lebensdauer der WP [Jahre]
CAPEX_spez = 700        # Spezifische Investitionskosten [€/kW_th]
subsidy_rate = 0.4      # Förderquote z.B. BEW (40 % der Investitionskosten) [-]
#discount_rate = 0.05    # Diskontierungszinssatz für LCOH-Berechnung [-]

# --- Wärmepumpe: Betriebsregeln (für spätere MIP-Erweiterung) ---
#Q_min = 5.0             # Mindestleistung bei Betrieb [MW_th]
                        # (WP läuft nur, wenn mind. 5 MW geliefert werden)
#t_min_off = 4           # Mindestabschaltdauer [h]
                        # (nach Abschalten mind. 4 h aus)

# --- Speicher ---
storage_cap = 500       # Maximale Speicherkapazität [MWh]
charge_max = 80         # Maximale Lade-/Entladeleistung [MW]
eta = 0.9               # Wirkungsgrad Speicher (Laden & Entladen) [-]

#-----------------------------------------------------------------------------

# Modell
model = pyo.ConcreteModel()

# Zeitindex im Modell
model.T = pyo.Set(initialize=T)

#-----------------------------------------------------------------------------
# Variablen
#-----------------------------------------------------------------------------

# --- Wärmepumpe ---

# Thermische Leistung der WP in Stunde t [MW_th]
# -> Wie viel Wärme liefert die WP in Stunde t?
# bounds: min. 0, max. Q_wp_max
model.Q_wp = pyo.Var(model.T, bounds=(0, Q_wp_max))

# Elektrische Leistungsaufnahme der WP in Stunde t [MW_el]
# -> Wie viel Strom verbraucht die WP in Stunde t?
# Zusammenhang: Q_wp = COP * P_wp  ->  P_wp = Q_wp / COP
model.P_wp = pyo.Var(model.T, bounds=(0, Q_wp_max / COP))

# An/Aus-Status der WP in Stunde t [-]
# -> 1 = WP läuft, 0 = WP ist aus
# HINWEIS: Binärvariable -> wird erst in Schritt 2 (MIP) aktiv genutzt
#          (Mindestlaufzeit, Mindestleistung, Abschaltregeln)
#          Jetzt schon definiert, damit die Struktur vollständig ist
# model.u = pyo.Var(model.T, within=pyo.Binary)

# --- Speicher ---

# Ladeleistung des Speichers in Stunde t [MW_th]
# -> Wie viel Wärme wird in Stunde t in den Speicher geladen?
model.charge = pyo.Var(model.T, bounds=(0, charge_max))

# Entladeleistung des Speichers in Stunde t [MW_th]
# -> Wie viel Wärme wird in Stunde t aus dem Speicher entnommen?
model.discharge = pyo.Var(model.T, bounds=(0, charge_max))

# State of Charge: Füllstand des Speichers in Stunde t [MWh]
# -> Wie voll ist der Speicher in Stunde t?
# bounds: min. 0 (nicht negativ), max. storage_cap
model.SOC = pyo.Var(model.T, bounds=(0, storage_cap))

# --- Slack ---

# Ungedeckter Wärmebedarf in Stunde t [MW_th]
# -> Wärme, die weder WP noch Speicher liefern können
# -> Wird später durch den Gaskessel gedeckt
# -> Macht das Modell lösbar (feasible), solange WP zu klein für Volllast
model.Q_slack = pyo.Var(model.T, bounds=(0, None))

#-----------------------------------------------------------------------------
# Zielfunktion (Bewertungsregel)
#-----------------------------------------------------------------------------

# Strafkosten für ungedeckten Wärmebedarf [€/MWh]
# Sehr hoch angesetzt, damit Solver Q_slack so klein wie möglich hält
penalty = 10000

# Ziel: Minimiere Stromkosten der WP + Strafkosten für ungedeckte Last
# price[t]   -> realer Strompreis in Stunde t [€/MWh_el]
# P_wp[t]    -> elektrische Leistungsaufnahme WP in Stunde t [MW_el]
# Q_slack[t] -> ungedeckter Wärmebedarf in Stunde t [MW_th]
#
# Stromkosten = Preis [€/MWh] * Leistung [MW] * 1 Stunde = [€]
def obj_rule(m):
    return sum(price[t] * m.P_wp[t] + penalty * m.Q_slack[t] for t in m.T)

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

#-----------------------------------------------------------------------------
# Constraints
#-----------------------------------------------------------------------------

# 1. Kopplung Q_wp und P_wp über COP
#    Q_wp[t] = COP * P_wp[t]
#    -> stellt sicher, dass beide Variablen konsistent sind

def cop_rule(m, t):
    return m.Q_wp[t] == COP * m.P_wp[t]

model.cop_constraint = pyo.Constraint(model.T, rule=cop_rule)

# 2. Wärmebilanz: Angebot == Nachfrage in jeder Stunde
#    Verfügbare Wärme (WP + Speicher entladen + Slack)
#    == Benötigte Wärme (Last + Speicher laden)

def heat_balance(m, t):
    return m.Q_wp[t] + m.discharge[t] + m.Q_slack[t] == demand[t] + m.charge[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# 3. Speicherdynamik: Füllstand in Stunde t
#    SOC[t] = SOC[t-1] + eta*charge[t] - (1/eta)*discharge[t]
#    -> Stunde 0: Speicher startet leer

def storage_rule(m, t):
    if t == 0:
        return m.SOC[t] == 0
    return m.SOC[t] == m.SOC[t-1] + eta * m.charge[t] - (1/eta) * m.discharge[t]

model.storage = pyo.Constraint(model.T, rule=storage_rule)

# 4. Speicherkapazität: SOC darf storage_cap nicht überschreiten
#    -> bereits durch bounds=(0, storage_cap) in der Variable gesichert
#    -> kein extra Constraint nötig

# 5. 100 % Strom aus Erneuerbaren/PV
#    Annahme: PV liefert konstant P_pv [MW_el]
#    Netzstrom = max(0, P_wp[t] - P_pv)
#    Linearisierung: P_net[t] >= P_wp[t] - P_pv
#                    P_net[t] >= 0

P_pv = 3.0              # Konstante PV-Leistung [MW_el] (Vereinfachung)

model.P_net = pyo.Var(model.T, bounds=(0, None))  # Netzstrom [MW_el]

def pv_rule(m, t):
    # Netzstrom = WP-Strombedarf abzüglich PV-Eigenversorgung
    return m.P_net[t] >= m.P_wp[t] - P_pv

model.pv_constraint = pyo.Constraint(model.T, rule=pv_rule)

# ----------------------------------------------------------------
# SCHRITT 2 (noch nicht aktiv - benötigt Binärvariable model.u):
#
# 6. Mindestleistung bei Betrieb: Q_wp[t] >= Q_min * u[t]
#    -> WP läuft nur, wenn mind. 5 MW_th geliefert werden
#
# 7. Mindestabschaltdauer: mind. 4 h aus nach Abschalten
#    -> komplexe Logik mit u[t], u[t-1]
#
# 8. Keine schnellen An/Aus-Wechsel
#    -> ebenfalls über u[t] geregelt

# ----------------------------------------------------------------
# Solver
# ----------------------------------------------------------------

solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

# ----------------------------------------------------------------
# Ergebnisse
# ----------------------------------------------------------------

Q_wp_res      = np.array([pyo.value(model.Q_wp[t])      for t in T])
P_wp_res      = np.array([pyo.value(model.P_wp[t])      for t in T])
charge_res    = np.array([pyo.value(model.charge[t])    for t in T])
discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
SOC_res       = np.array([pyo.value(model.SOC[t])       for t in T])
Q_slack_res   = np.array([pyo.value(model.Q_slack[t])   for t in T])

# ----------------------------------------------------------------
# KPIs (Results)
# ----------------------------------------------------------------

# 1. Heat Pump
# Total electricity costs (OPEX)
#    Strompreis [€/MWh] * elektr. Leistung [MW] * 1h = [€]
opex_total = sum(price[t] * P_wp_res[t] for t in T)

# 2. Operating hours
#    Anzahl Stunden, in denen die WP Wärme liefert (Q_wp > 0)
operating_hours = np.sum(Q_wp_res > 0.01)  # 0.01 als Toleranz gegen Rundungsfehler

# Stunden auf Volllast
wp_hours_fullload = np.sum(Q_wp_res >= Q_wp_max * 0.99) 

# Coverage rate
#    Wie viel % der gesamten Wärmelast deckt die WP?
coverage = np.sum(Q_wp_res) / np.sum(demand) * 100

# MWh ins Netz gespeist
wp_annual_heat    = np.sum(Q_wp_res)

# 2. Slack (Backup-Bedarf)
#    Wie viel Wärme konnte weder WP noch Speicher liefern?
slack_total = np.sum(Q_slack_res)
slack_share = slack_total / np.sum(demand) * 100

# 5. Speicher
storage_charged    = np.sum(charge_res)                     # MWh geladen gesamt
storage_discharged = np.sum(discharge_res)                  # MWh entladen gesamt
storage_hours      = np.sum(charge_res > 0.01)              # Stunden aktiv geladen

# 6. Gesamtbilanz
total_demand = np.sum(demand)

print(f"\n--- Wärmepumpe ---")
print(f"Total electricity costs (OPEX): {opex_total:,.0f} €")
print(f"Operating hours: {operating_hours} h von {len(T)} h")
print(f"  Stunden auf Volllast:        {wp_hours_fullload} h/Jahr")
print(f"Coverage rate WP: {coverage:.1f} %")
print(f"  Wärme ins Netz gespeist:     {wp_annual_heat:,.0f} MWh/Jahr")

print(f"\n--- Slack / Backup (Gaskessel-Platzhalter) ---")
print(f"Unmet demand (Slack/Backup): {slack_total:,.0f} MWh  ({slack_share:.1f} % der Last)")

print(f"\n--- Speicher ---")
print(f"  Geladene Energie gesamt:     {storage_charged:,.0f} MWh/Jahr")
print(f"  Entladene Energie gesamt:    {storage_discharged:,.0f} MWh/Jahr")
print(f"  Ladestunden:                 {storage_hours} h/Jahr")
print(f"  Max. Füllstand erreicht:     {np.max(SOC_res):.1f} MWh  (von {storage_cap} MWh)")

print(f"\n--- Gesamtbilanz ---")
print(f"  Gesamter Wärmebedarf:        {total_demand:,.0f} MWh/Jahr")
print(f"  Davon WP:                    {wp_annual_heat/total_demand*100:.1f} %")
print(f"  Davon Speicher:              {np.sum(discharge_res)/total_demand*100:.1f} %")
print(f"  Davon Backup/Slack:          {slack_share:.1f} %")

# Dauerlinie sortieren
sorted_idx = np.argsort(-demand)

demand_sorted    = demand[sorted_idx]
wp_sorted        = Q_wp_res[sorted_idx]
discharge_sorted = discharge_res[sorted_idx]
slack_sorted     = Q_slack_res[sorted_idx]

# ----------------------------------------------------------------
# Plots
# ----------------------------------------------------------------

# Plot 1: Dauerlinie (sortiert nach Last)
plt.figure()
plt.plot(demand_sorted,    label="Wärmebedarf")
plt.plot(wp_sorted,        label="Wärmepumpe (thermisch)")
plt.plot(discharge_sorted, label="Speicher Entladung")
plt.plot(slack_sorted,     label="Backup / Slack")
plt.xlabel("Stunden (sortiert nach Last)")
plt.ylabel("Leistung [MW_th]")
plt.title("Dauerlinie: Einsatz von WP und Speicher")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Speicher Lade-/Entladevorgänge + Füllstand (SOC)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(charge_res,    label="Laden",    color="green")
ax1.plot(discharge_res, label="Entladen", color="red")
ax1.set_ylabel("Leistung [MW_th]")
ax1.set_title("Speicher: Lade-/Entladevorgänge und Füllstand")
ax1.legend()
ax1.grid()

ax2.plot(SOC_res, label="State of Charge", color="orange")
ax2.set_ylabel("Energie [MWh]")
ax2.set_xlabel("Zeit [h]")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

# Plot 3: Zeitreihe WP, Speicher und Wärmebedarf
plt.figure()
plt.plot(demand,        label="Wärmebedarf",       color="black",  linewidth=1.5)
plt.plot(Q_wp_res,      label="Wärmepumpe (Q_wp)", color="blue")
plt.plot(discharge_res, label="Speicher Entladung", color="orange")
plt.plot(Q_slack_res,   label="Backup / Slack",     color="red",   linestyle="--")
plt.xlabel("Zeit [h]")
plt.ylabel("Leistung [MW_th]")
plt.title("Zeitreihe: Einsatz von WP und Speicher")
plt.legend()
plt.grid()
plt.show()
