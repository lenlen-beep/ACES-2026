#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:12:29 2026

@author: matsbeyer
"""

import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

"Einlesen-------------------------------"
load_file  = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/district_heating_data_Flensburg_2017.xlsx"
pv_file    = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/ninja_pv_54.7833_9.4333_uncorrected.csv"
price_file = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/Gro_handelspreise_202401010000_202501010000_Stunde.xlsx"

# Wärmelast
df_load = pd.read_excel(load_file, skiprows=1, header=0)
df_load.columns = ['Datum', 'Wärmeleistung in MW']
df_load['Datum'] = pd.to_datetime(df_load['Datum'])

# PV-Einstrahlungsdaten (electricity in kW pro kWp, normiert 0–1)
df_pv = pd.read_csv(pv_file, skiprows=3, header=0, parse_dates=['time'])
df_pv = df_pv[['time', 'electricity']].rename(columns={'time': 'Datum', 'electricity': 'PV_kW'})
df_pv['Datum'] = pd.to_datetime(df_pv['Datum'], utc=True).dt.tz_localize(None)

# Strompreise DE/LU [€/MWh]
df_price = pd.read_excel(price_file, skiprows=9, header=0)
df_price = df_price[['Datum von', 'Deutschland/Luxemburg [€/MWh]']].rename(
    columns={'Datum von': 'Datum', 'Deutschland/Luxemburg [€/MWh]': 'Preis_EUR_MWh'}
)
df_price['Datum'] = pd.to_datetime(df_price['Datum'], dayfirst=True)
df_price['Preis_EUR_MWh'] = pd.to_numeric(df_price['Preis_EUR_MWh'], errors='coerce')

T = range(len(df_load))

demand = df_load['Wärmeleistung in MW'].values
pv     = df_pv['PV_kW'].values[:len(T)]
price  = df_price['Preis_EUR_MWh'].values[:len(T)]
"---------------------------------------"

"Parameter------------------------------"
# Parameter
P_wp_max      = 350     # [MW]
P_pv_max_ub   = 500     # Obere Schranke PV-Anlagengröße [MW]
verguetung    = 30      # Einspeisevergütung [€/MWh]
eta_pv        = 0.85    # System Efficiency PV [-]
netzentgelt   = 70      # Netzentgelte [€/MWh]

# PV Wirtschaftlichkeit
capex         = 800_000 # Investitionskosten [€/MW]
opex          = 0.02    # Jährliche Betriebskosten [% des CAPEX /a]
lifetime      = 20      # Projektlaufzeit [a]
discount_rate = 0.05    # Diskontierungsrate [-]

# Annuitätenfaktor
annuity_factor = (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1)
annual_cost_per_mw = capex * (annuity_factor + opex)  # [€/MW/a]
"---------------------------------------"

"Pyomo-Variabeln------------------------"
# Modell
model = pyo.ConcreteModel()

model.T = pyo.Set(initialize=T)

# Variablen
model.P_wp    = pyo.Var(model.T, bounds=(0, P_wp_max))
model.P_pv    = pyo.Var(model.T, bounds=(0, None))        # genutzte PV-Leistung [MW]
model.P_feed  = pyo.Var(model.T, bounds=(0, None))        # eingespeiste PV-Leistung [MW]
model.P_grid  = pyo.Var(model.T, bounds=(0, None))        # Strombezug aus Netz [MW]
model.P_pv_max = pyo.Var(bounds=(0, P_pv_max_ub))         # optimale PV-Anlagengröße [MW]
"---------------------------------------"

"Zielfunktionen-------------------------"
# Zielfunktion: Investitionskosten + Netzstromkosten - Einspeiseerlöse
def obj_rule(m):
    return (
        annual_cost_per_mw * m.P_pv_max
        + sum(
            (price[t] + netzentgelt) * m.P_grid[t]
            - verguetung * m.P_feed[t]
            for t in m.T
        )
    )

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


# Wärmebilanz: WP-Leistung deckt Wärmebedarf
def heat_balance(m, t):
    return m.P_wp[t] == demand[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Strombilanz: WP wird aus PV + Netz versorgt
def power_balance(m, t):
    return m.P_wp[t] == m.P_pv[t] + m.P_grid[t]

model.power_balance = pyo.Constraint(model.T, rule=power_balance)

# Gesamte PV-Nutzung (Eigenverbrauch + Einspeisung) begrenzt durch Anlagengröße und Profil
def pv_limit(m, t):
    return m.P_pv[t] + m.P_feed[t] <= pv[t] * m.P_pv_max * eta_pv

model.pv_limit = pyo.Constraint(model.T, rule=pv_limit)
"---------------------------------------"

# Solver
solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

# Ergebnisse
P_pv_max_res = pyo.value(model.P_pv_max)
P_wp_res     = np.array([pyo.value(model.P_wp[t])   for t in T])
P_pv_res     = np.array([pyo.value(model.P_pv[t])   for t in T])
P_feed_res   = np.array([pyo.value(model.P_feed[t]) for t in T])
P_grid_res   = np.array([pyo.value(model.P_grid[t]) for t in T])
demand = np.array(demand)

# Konsolenausgaben
kosten    = sum((price[t] + netzentgelt) * P_grid_res[t] for t in T)
einnahmen = sum(verguetung * P_feed_res[t] for t in T)
ersparnis = sum((price[t] + netzentgelt) * P_pv_res[t] for t in T)

print(f"Optimale PV-Anlagengröße:     {P_pv_max_res:>12,.2f} MW")
print(f"Ausgaben für Netzstrom:       {kosten:>12,.2f} €")
print(f"Einnahmen durch Einspeisung:  {einnahmen:>12,.2f} €")
print(f"Ersparnis durch Eigenverbrauch:{ersparnis:>11,.2f} €")

# LCOE berechnen (nach Optimierung, mit optimaler Anlagengröße)
lcoe = annual_cost_per_mw / (sum(pv) * eta_pv)  # [€/MWh]

# Plot: Täglicher Strompreis vs. LCOE
tage = len(T) // 24
daily_price = np.array([
    np.mean(price[t*24:(t+1)*24] + netzentgelt) for t in range(tage)
])

plt.figure()
plt.plot(daily_price, label="Ø Strompreis inkl. Netzentgelt [€/MWh]")
plt.axhline(lcoe, color='red', linestyle='--', label=f"LCOE PV = {lcoe:.1f} €/MWh")
plt.xlabel("Tag")
plt.ylabel("Preis [€/MWh]")
plt.title("Täglicher Strompreis vs. LCOE der PV-Anlage")
plt.legend()
plt.grid()
plt.show()

# Plot: Zeitreihe
plt.figure()
plt.plot(P_pv_res,   label="PV-Eigenverbrauch")
plt.plot(P_feed_res, label="PV-Einspeisung")
plt.plot(P_grid_res, label="Netzstrom")
plt.plot(P_wp_res,   label="WP Strombedarf", linestyle='--')
plt.xlabel("Zeit [h]")
plt.ylabel("Leistung [MW]")
plt.title("Zeitreihe: Stromversorgung der Wärmepumpe")
plt.legend()
plt.grid()
plt.show()
