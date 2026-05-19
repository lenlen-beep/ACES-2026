import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from oemof.demand import bdew

# Import Meteostat library and dependencies
from datetime import datetime
import meteostat as ms
from pathlib import Path


# --------------------------------------------------
# EIngabe Gesamtlast in MW
# --------------------------------------------------

rated_load = 3

# dena Gebäudereport 2024 (Wohngebäudebestand) 
# verwendete Aufteileung: 66% EFH, 16% ZFH, 15% MFH

load_EFH = rated_load * 0.66 + rated_load * 0.16
# load_ZFH = rated_load * 0.16 ZFH gibt es in BDEW nicht
load_MFH = rated_load * 0.15

def load_temperature_data(
    lat,
    lon,
    year,
    reload_data=False,
    cache_dir="weather_cache"
):

    # --------------------------------------------------
    # Cache-Ordner
    # --------------------------------------------------

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Dateiname
    # --------------------------------------------------

    filename = f"weather_{lat:.3f}_{lon:.3f}_{year}.csv"

    filepath = cache_path / filename

    # --------------------------------------------------
    # Cache prüfen
    # --------------------------------------------------

    if filepath.exists() and not reload_data:

        print("Lade Wetterdaten aus Cache ...")

        df = pd.read_csv(
            filepath,
            index_col=0,
            parse_dates=True
        )

    else:

        print("Lade Wetterdaten von Meteostat ...")

        # --------------------------------------------------
        # Nächste Station finden
        # --------------------------------------------------

        nearby = ms.stations.nearby(ms.Point(lat, lon), limit=1)

        station_id = nearby.index[0]

        print(f"\nVerwendete Station: {station_id}")
        
        # --------------------------------------------------
        # Zeitraum
        # --------------------------------------------------

        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59)

        # --------------------------------------------------
        # Wetterdaten laden
        # --------------------------------------------------

        ts = ms.hourly(
            station_id,
            start,
            end
        )

        df = ts.fetch()

        # --------------------------------------------------
        # Lokal speichern
        # --------------------------------------------------

        df.to_csv(filepath)

        print(f"\nGespeichert unter:\n{filepath}")

    # --------------------------------------------------
    # Temperatur extrahieren
    # --------------------------------------------------

    temperature = df["temp"]

    return temperature, df


# ======================================================
# Beispiel
# ======================================================

temperature, weather_df = load_temperature_data(
    lat=54.78,
    lon=9.43,
    year=2024,
    reload_data=False
)

print(temperature.head())


# --------------------------------------------------
# Zeitindex
# --------------------------------------------------

index = pd.date_range(
    start="2024-01-01 00:00",
    end="2024-12-31 23:00",
    freq="1h"
)

# Sicherheitscheck:
temperature = temperature.reindex(index)

# Fehlende Werte interpolieren
temperature = temperature.interpolate()


# --------------------------------------------------
# Gebäude definieren
# --------------------------------------------------

buildings = [
    {
        "name": "EFH",
        "shlp_type": "EFH",
        "annual_heat_demand": load_EFH * 1e6
    },
    {
        "name": "MFH",
        "shlp_type": "MFH",
        "annual_heat_demand": load_MFH * 1e6
    }
]

# --------------------------------------------------
# Lastprofile erzeugen
# --------------------------------------------------

profiles = {}

for b in buildings:

    is_residential = b["shlp_type"] in ("EFH", "MFH")
    model = bdew.HeatBuilding(
        index,
        temperature=temperature,
        annual_heat_demand=b["annual_heat_demand"],
        shlp_type=b["shlp_type"],
        wind_class=0,
        building_class=1 if is_residential else 0
    )

    profiles[b["name"]] = model.get_bdew_profile()

# --------------------------------------------------
# Gesamtlast
# --------------------------------------------------

total_load = sum(profiles.values())

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(12,5))

for name, profile in profiles.items():
    plt.plot(profile, label=name, alpha=0.7)

plt.plot(
    total_load,
    label="Total",
    linewidth=2
)

plt.legend()
plt.grid()
plt.title("BDEW Heat Profiles")
plt.show()

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

# Nachfrage in einzelnen df
demand = df_load['Wärmeleistung in MW'].values 

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


# Variablen

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


# Zielfunktion (Bewertungsregel)

# Allgemein
#def name(model):
#    return ...
#model.obj = pyo.Objective(rule=name, sense=pyo.minimize)
#                                          =pyo.maximize)

#  hier: Stromkosten minimieren
def obj_rule(m):
    # Preis Strom [t] * Leistung WP [t]
    return sum(price[t] * m.P_wp[t] for t in m.T)

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
    return COP * m.P_wp[t] + m.discharge[t] == demand[t] + m.charge[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Speicher-Dynamik
def storage_rule(m, t):
    if t == 0:
        #Starte mit leerem Speicher
        return m.SOC[t] == 0
    # SOC immer gleich dem SOC aus vorherigen Zeitschritt + charge oder - discharge
    return m.SOC[t] == m.SOC[t-1] + eta*m.charge[t] - (1/eta)*m.discharge[t]

model.storage = pyo.Constraint(model.T, rule=storage_rule)

# Solver
solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

# Ergebnisse 
P_wp_res = np.array([pyo.value(model.P_wp[t]) for t in T])
charge_res = np.array([pyo.value(model.charge[t]) for t in T])
discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
demand = np.array(demand)

# Dauerlinie sortieren
sorted_idx = np.argsort(-demand)

demand_sorted = demand[sorted_idx]
wp_sorted = (COP * P_wp_res)[sorted_idx]
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
plt.plot(COP * P_wp_res, label="Wärmepumpe")
plt.plot(discharge_res, label="Speicher Entladung")
plt.plot(demand, label="Wärmebedarf")
plt.xlabel("Zeit [h]")
plt.ylabel("Leistung [MW]")
plt.title("Zeitreihe: Einsatz von WP und Speicher")
plt.legend()
plt.grid()
plt.show()