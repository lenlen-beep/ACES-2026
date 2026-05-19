import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt


"""
Optimierung eines Fernwärmesystems mit Wärmepumpe und thermischem Pufferspeicher.

Das Modell minimiert die Strom- und Speicherkosten unter Einhaltung der
Wärmeversorgung. Die Wärmepumpe kann Wärme direkt zur Lastdeckung bereitstellen
oder zeitlich flexibel in einen Speicher einspeichern.

Die Speicherdynamik wird über den Ladezustand (State of Charge, SOC) beschrieben.
Der SOC verändert sich in jedem Zeitschritt durch Be- und Entladung sowie
thermische Speicherverluste:

SOC(t) = SOC(t-1) + charge(t) - discharge(t) - losses

Die maximal speicherbare Energiemenge ergibt sich aus Speichervolumen,
Wassereigenschaften und Temperaturhub zwischen Vor- und Rücklauf.

Methodik:
- Zeitreihenbasierte Wärmebedarfsdaten
- Lineares Optimierungsmodell mit Pyomo
- Wärmebilanz und Speicherdynamik als Nebenbedingungen
- Kostenminimierung der Wärmepumpe und Speicherinvestition
"""


#-------------------------------------------------------------------------
# Einlesen Fernwärmedaten (Flensburg 2017, dummy)
#-------------------------------------------------------------------------

load_file  = r"src/Testprojects/district_heating_data_Flensburg_2017.xlsx"
df_load = pd.read_excel(load_file, skiprows=0, header=0)

df_load.columns = ['Datum', 'Wärmeleistung in MW']
df_load['Datum'] = pd.to_datetime(df_load['Datum'])

T = range(len(df_load))

demand_scale = 0.01 # da Fernwärmedaten etwas groß

# Nachfrage in einzelnen df
demand = df_load['Wärmeleistung in MW'].values * demand_scale #skaliert
heat_supply = demand.sum()


#-------------------------------------------------------------------------
# Einlesen Strompreise (Großhandelsstrompreise 2024, dummy)
#-------------------------------------------------------------------------

price_file  = r"src/Testprojects/Gro_handelspreise_202401010000_202501010000_Stunde.xlsx"
df_price = pd.read_excel(price_file, skiprows=9, header=0, usecols=['Datum von', 'Deutschland/Luxemburg [€/MWh]'])

df_price.columns = ['Datum von', 'Deutschland/Luxemburg [€/MWh]']

# Spalte umbenennen
df_price.rename(columns={'Datum von': 'Datum'}, inplace=True)

df_price['Datum'] = pd.to_datetime(
    df_price['Datum'],
    format='%d.%m.%Y %H:%M'
)

# Spalte umbenennen
df_price.rename(columns={'Datum von': 'Datum'}, inplace=True)

# 29.02. entfernen (2024 Schaltjahr)
df_price = df_price[
    ~((df_price['Datum'].dt.month == 2) & (df_price['Datum'].dt.day == 29))
]

# Daten löschen (29.03.2024, 31.03.2024), da Zeitumstellung (--> 2 Uhr fehlt und unterschiedlicher Tag)
df_price = df_price[
    ~((df_price['Datum'].dt.month == 3) & (df_price['Datum'].dt.day == 26) & (df_price['Datum'].dt.hour == 2))
]
df_load = df_load[
    ~((df_load['Datum'].dt.month == 3) & (df_load['Datum'].dt.day == 31) & (df_load['Datum'].dt.hour == 2))
]

price = df_price['Deutschland/Luxemburg [€/MWh]'].values 


#-------------------------------------------------------------------------
# Anlagensparameter
#-------------------------------------------------------------------------

# --- Zinsrechung für Investitionen ---

n = 20 #a Ammoritsationszeitraum
r = 0.05 #Zins
annuity_factor = r * (1+r)**n / ((1+r)**n - 1)


# --- Netzparameter --> max 95 °C für einfache Pufferspeicher (-->Constraint!) ---

VLT = 80 #°C
RLT = 55 #°C
cp_W = 4.187 #kJ/kgK
rho_W = 1000 #kg/m3
delta_T = VLT - RLT #K

# Warnung, wenn Vorlauftemperatur zu hoch ist (Speicher ungeeignet)
if VLT > 95:
    import warnings
    warnings.warn("VLT > 95°C: Pufferspeicher sind dafür nicht geeignet. Bitte Temperaturgrenzen prüfen.", UserWarning)


# --- Parameter Speicher ---

storage_cap = 500 #m3 --> initial condition Speicherkapazität
max_charge_rate = max_discharge_rate = 0.25 # % der Speicherkapazität pro Zeitschritt (z.B. 0.25 = 25% der Kapazität kann pro Stunde geladen werden)
p_loss = 1.67/(24*1000) #MWh Wärmeverlust --> Technikkatalog MITTELWERT --> PRÜFEN
SOC_init = 0.2 # % initialer Ladezustand (Anfang u. Ende)

# Investitionskosten Speicher (linearisiert: Offset + spezifische Kosten * Kapazität) --> umgerechnet auf jährliche Kosten mit Annuitätenfaktor
# --> Langfristig vielleicht Abschätzung (größe) und entsprechende Gerade nehmen!
storage_invest_offset = 196675 #€ --> aus Technikkatalog (Offset) 
storage_specific_cost = 86.139/1000 #€/L --> aus Technikkatalog (linearisiert)

# Pumpe (Penalty für nicht gleichzeitges laden/entladen)
g = 9.81 #m/s2
h = 5 #m Förderhöhe (Schätzung)
eta_p = 0.9 #Pumpenwirkungsgrad
P_pump = (g * h) / (eta_p * cp_W*1000 *delta_T) #MW --> aus Netzparametern berechnen VAR


# --- Parameter WP ---

COP = 3.5
Pth_wp_max = 3.5 #MW


#------------------------------------------------------------------------
# Pyomo-Modellierung
#------------------------------------------------------------------------

# Modell
model = pyo.ConcreteModel()

# Zeitindex im Modell
model.T = pyo.Set(initialize=T)


# --- Variablen ---

# Wärmepumpe
model.P_wp = pyo.Var(model.T, bounds=(0, Pth_wp_max)) #MW

# Speicher
model.charge = pyo.Var(model.T, bounds=(0, None)) #MW
model.discharge = pyo.Var(model.T, bounds=(0, None)) #MW
model.SOC = pyo.Var(model.T, bounds=(0, None)) #kWh
model.storage_capacity = pyo.Var(bounds=(0, None), initialize=storage_cap) #m3


# --- Zielfunktion (Bewertungsregel) ---

# Stromkosten minimieren
def obj_rule(m):
    # Preis Strom [t] * Leistung WP [t] + Investkosten Speicher (linearisiert: offset + spezifische Kosten * Kapazität) --> umgerechnet auf jährliche Kosten mit Annuitätenfaktor
    return sum(price[t] * (m.P_wp[t]/COP) for t in m.T) + (storage_invest_offset + storage_specific_cost * (m.storage_capacity)) * annuity_factor + P_pump * sum(price[t] * (m.charge[t] + m.discharge[t]) for t in m.T)

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


# --- Constraints ---

# Wärmebilanz
def heat_balance(m, t):
    #Verfügbare Wärmeleistung (P_WP + P_SP) == Benötigte Wärmeleistung (Last + P_SP)
    return m.P_wp[t] + m.discharge[t] == demand[t] + m.charge[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

def storage_volume_to_MWh(vol_L):
    # vol_L: liters
    # vol_m3 = L 
    # mass_kg = vol_m3 * rho_W
    # energy_kJ = mass_kg * cp_W * delta_T
    # energy_kWh = energy_kJ / 3600
    # energy_MWh = energy_kWh / 1000
    return (vol_L) * rho_W * cp_W * delta_T / (3600.0) #MWh

# Speicher-Dynamik
def storage_rule(m, t):
    storage_MWh = storage_volume_to_MWh(m.storage_capacity)
    p_loss_MWh = p_loss #kWh --> MWh
    if t == 0:
        #return m.SOC[t] == 0
        return m.SOC[t] == 0 #SOC_init * storage_MWh #x% initialer Ladezustand (Anfang)
    
    elif t == T[-1]:
        return m.SOC[t] == 0 #SOC_init * storage_MWh #x% initialer Ladezustand (Ende)
    
    # SOC immer gleich dem SOC aus vorherigen Zeitschritt + charge oder - discharge
    return m.SOC[t] == m.SOC[t-1] + m.charge[t] - m.discharge[t] -p_loss_MWh

model.storage = pyo.Constraint(model.T, rule=storage_rule)

# TODO:Der SPeicher kann bei SOC = 0 keine Wärme verlieren (theoretisch aber schon, da 55°C RLT) --> beachten :)

def soc_capacity_limit(m, t):
    storage_MWh = storage_volume_to_MWh(m.storage_capacity)
    return m.SOC[t] <= storage_MWh

model.soc_capacity_limit = pyo.Constraint(model.T, rule=soc_capacity_limit)

def charge_power_limit_storage(m, t):
    storage_MWh = storage_volume_to_MWh(m.storage_capacity)

    return m.charge[t] <= max_charge_rate * storage_MWh

def charge_power_limit_hp(m, t):
    return m.charge[t] <= Pth_wp_max

def negative_price_discharge_restrict(m, t):
    if price[t] < 0:
        return m.discharge[t] == 0 # Bei negativen Preisen, entladen verboten.
    else:
        return pyo.Constraint.Skip
    
model.negative_price_discharge_restrict = pyo.Constraint(model.T, rule=negative_price_discharge_restrict)


def discharge_power_limit(m, t):
    storage_MWh = storage_volume_to_MWh(m.storage_capacity)
    return m.discharge[t] <= max_discharge_rate * storage_MWh

model.charge_power_limit_storage = pyo.Constraint(model.T, rule=charge_power_limit_storage)
model.charge_power_limit_hp = pyo.Constraint(model.T, rule=charge_power_limit_hp)

# Solver
solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

# Ergebnisse 
P_wp_res = np.array([pyo.value(model.P_wp[t]) for t in T])
charge_res = np.array([pyo.value(model.charge[t]) for t in T])
discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
SOC_res = np.array([pyo.value(model.SOC[t]) for t in T])
demand = np.array(demand)
storage_cap_res = pyo.value(model.storage_capacity)

print(f'Die Speichergröße beträgt {storage_cap_res} m3 bzw. {storage_cap_res * cp_W * delta_T /(3600)} kWh')


# --- PLOTS ---

# # Dauerlinie sortieren
sorted_idx = np.argsort(-demand)

demand_sorted = demand[sorted_idx]
wp_sorted = (P_wp_res)[sorted_idx]
discharge_sorted = discharge_res[sorted_idx]

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

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

# oben: Laden / Entladen
ax1.plot(charge_res, label="Laden")
ax1.plot(discharge_res, label="Entladen")
ax1.set_ylabel("Leistung [MW]")
ax1.set_title("Speicher Lade- und Entladevorgänge")
ax1.legend()
ax1.grid(True)

# unten: SOC
ax2.plot(SOC_res, color="tab:green", label="SOC")
if storage_cap_res is not None and storage_cap_res > 0:
    storage_MWh = storage_volume_to_MWh(storage_cap_res)
    ax2.plot(np.full(len(SOC_res), storage_MWh), '--', color="gray", label="Kapazität (MWh)")
ax2.set_xlabel("Zeit [h]")
ax2.set_ylabel("Energie [MWh]")
ax2.set_title("State of Charge (SOC)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
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

# --- Plot SOC ---
plt.figure()
plt.plot(SOC_res, label="SOC (model units)")
# if storage_capacity is set, show capacity line and fraction
if storage_cap_res is not None and storage_cap_res > 0:
    storage_MWh = storage_volume_to_MWh(storage_cap_res)
    plt.plot(np.full(len(SOC_res), storage_MWh), '--', label="Capacity (MWh)")
    plt.figure()
    plt.plot(SOC_res / storage_MWh, label="SOC / Capacity")
    plt.xlabel("Time [h]")
    plt.ylabel("Fraction of capacity")
    plt.title("SOC as fraction of capacity")
    plt.legend()
    plt.grid()

plt.xlabel("Time [h]")
plt.ylabel("SOC (kWh) / model units")
plt.title("State of Charge (SOC) over time")
plt.legend()
plt.grid()
plt.show()

