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

load_file  = r"src/Testprojects/district_heating_data_Flensburg_2017.xlsx"

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

demand_scale = 0.08

# Nachfrage in einzelnen df
demand = df_load['Wärmeleistung in MW'].values * demand_scale #skaliert
heat_supply = demand.sum()

# Strompreis (synthetisch)
price = {t: 50 + 20*np.sin(t/24) for t in T}

# Parameter
COP = 3.5
P_wp_max = 150
storage_cap = 500
charge_max = 80
eta = 0.9


# Modell
model = pyo.ConcreteModel()

# Zeitindex im Modell
model.T = pyo.Set(initialize=T)


# --- Variablen ---

# Allgemein
#model.x = pyo.Var(model.T, bounds=(min, max)) -- Wertebereich
#model.x = pyo.Var(model.T) -- ohne Begrenzung
#model.x = pyo.Var(model.T, bounds=(0, None)) -- nur >= 0
#model.x = pyo.Var(model.T, bounds=(None, max)) -- nur max begrenzt
#model.x = pyo.Var(model.T, within=pyo.Integers) -- ganzzahlig
#model.x = pyo.Var(model.T, within=pyo.Binary) -- an/aus

model.P_wp = pyo.Var(model.T, bounds=(0, P_wp_max))
model.charge = pyo.Var(model.T, bounds=(0, charge_max))
model.discharge = pyo.Var(model.T, bounds=(0, charge_max))
model.SOC = pyo.Var(model.T, bounds=(0, storage_cap))


# --- Zielfunktion (Bewertungsregel) ---

# Allgemein
#def name(model):
#    return ...
#model.obj = pyo.Objective(rule=name, sense=pyo.minimize)
#                                          =pyo.maximize)

#  hier: Stromkosten minimieren
def obj_rule(m):
    # Preis Strom [t] * Leistung WP [t]
    return sum(price[t] * (m.P_wp[t]/COP) for t in m.T)

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Constraints

# Allgemein: Regeln, die eingehalten werden müssen
# Immer:
#       Gleichung: A==B
#       Ungleichung: A<=B, A>=B

#def x(m, t): (meistens model und Zeitschritt, nicht immer)
#    return ...

#model.xx = pyo.Constraint(model.T, rule=x)

# Wärmebilanz
def heat_balance(m, t):
    #Verfügbare Wärmeleistung (COP*P_WP + P_SP) == Benötigte Wärmeleistung (Last + P_SP)
    return m.P_wp[t] + m.discharge[t] == demand[t] + m.charge[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Speicher-Dynamik
def storage_rule(m, t):
    if t == 0:
        #Starte mit leerem Speicher
        return m.SOC[t] == 0
    # SOC immer gleich dem SOC aus vorherigen Zeitschritt + charge oder - discharge
    return m.SOC[t] == m.SOC[t-1] + eta*m.charge[t] - (1/eta)*m.discharge[t]
#TODO: 

model.storage = pyo.Constraint(model.T, rule=storage_rule)

def soc_capacity_limit(m, t):
    return m.SOC[t] <= m.storage_capacity * cp_W * delta_T / 3600

model.soc_capacity_limit = pyo.Constraint(model.T, rule=soc_capacity_limit)

def charge_power_limit(m, t):
    return m.charge[t] <= 0.25 * m.storage_capacity * cp_W * delta_T / 3600 #kW --> aus m_dot_max und Netzparametern berechnen

def discharge_power_limit(m, t):
    return m.discharge[t] <= 0.25 * m.storage_capacity * cp_W * delta_T / 3600

model.charge_power_limit = pyo.Constraint(model.T, rule=charge_power_limit)
model.discharge_power_limit = pyo.Constraint(model.T, rule=discharge_power_limit)
# Solver
solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

# Ergebnisse 
P_wp_res = np.array([pyo.value(model.P_wp[t]) for t in T])
charge_res = np.array([pyo.value(model.charge[t]) for t in T])
discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
demand = np.array(demand)
storage_cap_res = pyo.value(model.storage_capacity)

print(f'Die Speichergröße beträgt {storage_cap_res} Liter bzw. {storage_cap_res * cp_W * delta_T /(3600*1000)} MWh')
# Dauerlinie sortieren
sorted_idx = np.argsort(-demand)

demand_sorted = demand[sorted_idx]
wp_sorted = (P_wp_res)[sorted_idx]
discharge_sorted = discharge_res[sorted_idx]

# Plot

#Sortiert Last WP u. Speicher
plt.figure()
plt.plot(demand_sorted, label="Wärmebedarf")
plt.plot(wp_sorted, label="Wärmepumpe")
plt.plot(discharge_sorted, label="Speicher Entladung")
plt.xlabel("Stunden (sortiert)")
plt.ylabel("Leistung [MW]")
plt.title("Dauerlinie mit Einsatz von WP und Speicher")
plt.legend()
plt.grid()
plt.show()

#Lade- Entladezyklus Speicher
plt.figure()
plt.plot(charge_res, label="Laden")
plt.plot(discharge_res, label="Entladen")
plt.xlabel("Zeit [h]")
plt.ylabel("Leistung [MW]")
plt.title("Speicher Lade- und Entladevorgänge")
plt.legend()
plt.grid()
plt.show()

# Dauerlinie, Last WP + Speicher
plt.figure()
plt.plot(P_wp_res, label="Wärmepumpe")
plt.plot(discharge_res, label="Speicher Entladung")
plt.plot(demand, label="Wärmebedarf")
plt.xlabel("Zeit [h]")
plt.ylabel("Leistung [MW]")
plt.title("Zeitreihe: Einsatz von WP und Speicher")
plt.legend()
plt.grid()
plt.show()