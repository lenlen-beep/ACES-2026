import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

"""
Optimierung eines Fernwärmesystems mit Wärmepumpe und Gaskessel.

Das Modell minimiert die Gesamtkosten (Strom + Gas) unter Einhaltung der
Wärmeversorgung. Der Optimizer wählt je Zeitschritt die günstigere Quelle.

Methodik:
- Daten: Zeitreihe der Wärmeleistung (Excel)
- Optimierung: Pyomo (lineares Optimierungsmodell)
- Ziel: Minimierung der Gesamtkosten (Strom WP + Gas Kessel)
- Nebenbedingungen: Wärmebilanz
"""

load_file = r"src/Testprojects/district_heating_data_Flensburg_2017.xlsx"

df_load = pd.read_excel(load_file, skiprows=1, header=0)
df_load.columns = ['Datum', 'Wärmeleistung in MW']
df_load['Datum'] = pd.to_datetime(df_load['Datum'])

# Zeitindex
T = range(len(df_load))

# Nachfrage
demand = df_load['Wärmeleistung in MW'].values

# Preise (synthetisch)
price     = {t: 50 + 20*np.sin(t/24)           for t in T}
gas_price = {t: 20 + 15*np.sin(t/24 + np.pi/3) for t in T}

# Parameter
COP              = 2
P_wp_max         = 150
P_gas_max        = 200
eta_gas_boiler   = 0.98


# Modell
model   = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)

# Variablen
model.P_wp  = pyo.Var(model.T, bounds=(0, P_wp_max))
model.P_gas = pyo.Var(model.T, bounds=(0, P_gas_max))

# Zielfunktion: Gesamtkosten Strom (WP) + Gas (Kessel)
def obj_rule(m):
    return sum(
        price[t]     * m.P_wp[t]  / COP           # Stromkosten WP
      + gas_price[t] * m.P_gas[t] / eta_gas_boiler # Gaskosten Kessel
        for t in m.T
    )

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Wärmebilanz: WP-Wärme + Kessel-Wärme == Bedarf
def heat_balance(m, t):
    return COP * m.P_wp[t] + m.P_gas[t] == demand[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Solver
solver  = pyo.SolverFactory('glpk')
results = solver.solve(model)

# Ergebnisse
P_wp_res  = np.array([pyo.value(model.P_wp[t])  for t in T])
P_gas_res = np.array([pyo.value(model.P_gas[t]) for t in T])
demand    = np.array(demand)

Q_wp  = COP * P_wp_res   # Wärmeleistung WP
Q_gas = P_gas_res        # Wärmeleistung Kessel

price_arr     = np.array([price[t]     for t in T])
gas_price_arr = np.array([gas_price[t] for t in T])

# Kosten
cost_wp  = price_arr     * P_wp_res  / COP
cost_gas = gas_price_arr * P_gas_res / eta_gas_boiler

print(f"Gesamtkosten WP:     {cost_wp.sum():,.0f} €")
print(f"Gesamtkosten Kessel: {cost_gas.sum():,.0f} €")
print(f"Gesamtkosten total:  {(cost_wp + cost_gas).sum():,.0f} €")

# --------------------------------------------------
# Plot 1: Zeitreihe WP + Gaskessel + Bedarf
# --------------------------------------------------
plt.figure(figsize=(14, 4))
plt.plot(Q_wp,   label=f"Wärmepumpe  ({Q_wp.sum():.0f} MWh)", color='steelblue')
plt.plot(Q_gas,  label=f"Gaskessel   ({Q_gas.sum():.0f} MWh)", color='tomato')
plt.plot(demand, label="Wärmebedarf", color='black', linewidth=0.8, alpha=0.6)
plt.xlabel("Zeit [h]")
plt.ylabel("Leistung [MW]")
plt.title("Zeitreihe: Einsatz WP und Gaskessel")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 2: Dauerlinie (sortiert nach Bedarf)
# --------------------------------------------------
sorted_idx   = np.argsort(-demand)
plt.figure(figsize=(14, 4))
plt.plot(demand[sorted_idx], label="Wärmebedarf",  color='black', linewidth=0.8)
plt.plot(Q_wp[sorted_idx],   label="Wärmepumpe",   color='steelblue')
plt.plot(Q_gas[sorted_idx],  label="Gaskessel",     color='tomato')
plt.xlabel("Stunden (sortiert nach Bedarf)")
plt.ylabel("Leistung [MW]")
plt.title("Dauerlinie: WP und Gaskessel")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 3: Preise + Einsatzentscheidung
# --------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(price_arr,     label="Strompreis [€/MWh]", color='steelblue', linewidth=0.8)
ax1.plot(gas_price_arr, label="Gaspreis [€/MWh]",   color='tomato',    linewidth=0.8)
ax1.set_xlabel("Zeit [h]")
ax1.set_ylabel("Preis [€/MWh]")
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.fill_between(range(len(T)), Q_wp,  alpha=0.2, color='steelblue', label='WP aktiv')
ax2.fill_between(range(len(T)), Q_gas, alpha=0.2, color='tomato',    label='Kessel aktiv')
ax2.set_ylabel("Wärmeleistung [MW]")
ax2.legend(loc='upper right')
plt.title("Preise und Einsatzentscheidung")
plt.tight_layout()
plt.show()
