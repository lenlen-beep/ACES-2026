from funcs.create_SPL import load_temperature_data, create_bdew_prifles, create_mean_german_building_loads
from funcs.plots import plot_bdew_profiles, plot_temperatures, plot_prices, plot_gas_prices, \
                        plot_charge_discharge_process, plot_energy_system_output_sorted, \
                        plot_load_w_components, plot_SOC, plot_pv, plot_seasonal_storage
from funcs.read_data import read_price_data, read_gas_price_data, read_pv_data
from funcs.energy_system_optimization import optimize_energy_system
from funcs.net_modelling import load_network_gpkg, build_graph, test_connectivity, create_pandapipes_network, \
                                load_example_buildings, export_res_pipe_gpkg, run_timeseries
from funcs.data_analysis.test_buildings import load_example_buildings

import pandapipes
import pandas as pd

df = load_example_buildings()

from funcs.read_data import read_parameters
parameters = read_parameters("src/ACES-2026/parameters.yaml")

import numpy as np

# TODO: load_example_buildings muss durch den richtigen ersetzt werden

# TODO: Gaspreise jetzt tariflich -> Einlesen der Gaspreise kann weg

# -------------------------------------------------
# Netzsimulation
# -------------------------------------------------

# Trassierung importieren
gdf = load_network_gpkg(
    path="src/ACES-2026/Data/Trassierung.gpkg",
    layer="Trassierung",
)

# Netz bauen
graph = build_graph(gdf)
test_connectivity(graph)

net, pipe_geoms, pipe_pairs= create_pandapipes_network(graph)

# Testrechnung
pandapipes.pipeflow(net)
print(net.res_circ_pump_pressure)
export_res_pipe_gpkg(net, pipe_geoms, pipe_pairs, path="src/ACES-2026/Data/res_pipe.gpkg")

# Gebäudedaten laden
buildings_df = load_example_buildings()  

# Zeitreihensimulation
result_df = run_timeseries(net, buildings_df)

print(result_df)

# Spitzenlast filtern
peak_idx  = result_df['mdot_kg_per_s'].idxmax()
peak_mass_flow = result_df.loc[peak_idx, 'mdot_kg_per_s']
peak_date = result_df.loc[peak_idx, 'Datum']
peak_load = peak_mass_flow * parameters['net_parameters']['cp'] * parameters['net_parameters']['delta_T'] 
print(f"Spitzenlast: {peak_load:.1f} mit Spitzenmassenstrom: {peak_mass_flow:.4f} kg/s  am  {peak_date}")

# Dauerlinie berechnen
result_df['load_kW'] = result_df['mdot_kg_per_s'] * parameters['net_parameters']['cp'] * parameters['net_parameters']['delta_T'] 

print(result_df)

# Für die Rohrdimensionierung: Spitzenlastreihe simulieren
peak_row = df.iloc[[peak_idx]]  # doppelte Klammer → DataFrame statt Series
print(peak_row)

peak_result_df = run_timeseries(net, peak_row)

export_res_pipe_gpkg(net, pipe_geoms, pipe_pairs, path="src/ACES-2026/Data/res_pipe_peak.gpkg")


# Nennlast des Wärmenetzes (Gesamt) in MWh (?)
load = result_df.set_index('Datum')['load_kW']

# Angabe über mögliche Abwärmequellen (z.B. Industrie, Rechenzentren)
max_waste_heat_capacity = 0 # MW
waste_heat_cost = 0 # Euro/MWh

# Kann PV und Saisonalspeicher gebaut werden?
usable_area = 0 # m2


# --------------------------------------------------
# Laden der Temperaturdaten
# --------------------------------------------------

temperature, weather_df, time_index, station_id = load_temperature_data(
    lat=54.78,
    lon=9.43,
    year=2019,
    reload_data=False
)


# Referenzindex für 2024-Daten (Preise, PV): Schaltjahr = 8784 h
ref_2024 = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=8784, freq="1h"))

# --------------------------------------------------
# Laden und aufbereiten der Strompreise
# --------------------------------------------------

electricity_price = read_price_data(
    path="src/ACES-2026/Data/",
    filename="Gro_handelspreise_202401010000_202501010000_Stunde.xlsx",
    load_data=ref_2024
)


# --------------------------------------------------
# Laden der Gaspreise                                      #!
# --------------------------------------------------

gas_price = read_gas_price_data(
    path="src/ACES-2026/Data/",
    filename="Historic_THE_DA_Pegas.xlsx",
    load_data=ref_2024
)


# --------------------------------------------------
# Laden der PV-Daten
# --------------------------------------------------

pv = read_pv_data(
    path="src/ACES-2026/Data/",
    filename="ninja_pv_54.7833_9.4333_corrected.csv",
    load_data=ref_2024
)

# --------------------------------------------------
# Schalttag entfernen + Wochenprofil synchronisieren
# --------------------------------------------------
# Strom-/Gaspreise und PV liegen auf 2024-Achse (8784 h, startet Mo).
# Load ist 2019-Netzberechnung (8760 h, startet Di).
#   1. Feb 29 aus allen drei entfernen → 8760 h
#   2. Strompreise um 24 h rotieren: Mo → Di = passt zu 2019-Wochenprofil

feb29 = ~((ref_2024.index.month == 2) & (ref_2024.index.day == 29))

electricity_price = electricity_price[feb29]
gas_price         = gas_price[feb29]
pv                = pv[feb29]

# Ersten Tag (Mo, 1.1.2024, 24 h) ans Ende → beide Preisreihen starten Di (= 2019)
electricity_price = np.concatenate([electricity_price[24:], electricity_price[:24]])
gas_price         = np.concatenate([gas_price[24:],         gas_price[:24]])


# --------------------------------------------------
# Energiesystemoptimierung
# --------------------------------------------------

results, result_df_heatpump, result_df_gas_boiler, result_df_charge, result_df_discharge, \
    result_df_SOC, result_storage_capacity, result_gas_boiler_capacity, result_pv, \
    result_pv_feed_in, result_pv_capacity, result_seasonal_charge, result_seasonal_discharge, \
    result_seasonal_soc, result_seasonal_capacity \
    = optimize_energy_system(load, electricity_price, gas_price, pv)


# --------------------------------------------------
# Plotting
# --------------------------------------------------

plot_prices(electricity_price, show_plot=True)
plot_gas_prices(gas_price, show_plot=True)
plot_temperatures(temperature, station_id, show_plot=True)

plot_energy_system_output_sorted(load, 
                                 result_df_heatpump, 
                                 result_df_discharge, 
                                 result_df_gas_boiler,
                                 result_seasonal_discharge, 
                                 show_plot=True)

plot_load_w_components(result_df_heatpump, 
                       result_df_discharge, 
                       result_df_gas_boiler, 
                       load,
                       result_seasonal_discharge, 
                       show_plot=True)

plot_charge_discharge_process(result_df_charge, result_df_discharge, result_df_SOC, result_storage_capacity, show_plot=True)
plot_SOC(result_df_SOC, result_storage_capacity, show_plot=True)
plot_pv(result_pv, result_pv_feed_in, result_pv_capacity, show_plot=True)
plot_seasonal_storage(result_seasonal_charge, result_seasonal_discharge, result_seasonal_soc, result_seasonal_capacity, show_plot=True)
