from funcs.create_SPL import load_temperature_data, create_bdew_prifles, create_mean_german_building_loads
from funcs.plots import plot_bdew_profiles, plot_temperatures, plot_prices, plot_charge_discharge_process, \
                        plot_energy_system_output_sorted, plot_load_w_components, plot_SOC
from funcs.read_data import read_price_data
from funcs.energy_system_optimization import optimize_energy_system


# -------------------------------------------------
# Lastinputs definieren
# -------------------------------------------------

# Nennlast des Wärmenetzes (Gesamt) in MW
rated_load = 3e3

# Lastverteilung nach Gebäudetyp nach dena Gebäudereport 2024 (Wohngebäudebestand)
load_EFH, load_MFH = create_mean_german_building_loads(rated_load)


# --------------------------------------------------
# Laden der Temperaturdaten
# --------------------------------------------------

temperature, weather_df, time_index, station_id = load_temperature_data(
    lat=54.78,
    lon=9.43,
    year=2024,
    reload_data=False
)


# --------------------------------------------------
# Erstellen der BDEW Lastprofile
# --------------------------------------------------

# profiles: Einzelne Lastprofile je Gebäudekatiegorien, total_load: Summe der Lastprofile
profiles, total_load, total_heat_supply = create_bdew_prifles(
    load_EFH, 
    load_MFH, 
    temperature, 
    time_index
)


# --------------------------------------------------
# Laden und aufbereiten der Strompreise
# --------------------------------------------------

electricity_price = read_price_data(
    path="src/ACES-2026/Data/",
    filename="Gro_handelspreise_202401010000_202501010000_Stunde.xlsx",
    load_data=total_load
)


# --------------------------------------------------
# Energiesystemoptimierung
# --------------------------------------------------


results, result_df_heatpump, result_df_charge, result_df_discharge, result_df_SOC, \
    result_storage_capacity = optimize_energy_system(total_load, electricity_price)


# --------------------------------------------------
# Plotting
# --------------------------------------------------

plot_prices(electricity_price, show_plot=True)
plot_temperatures(temperature, station_id, show_plot=True)
plot_bdew_profiles(profiles, total_load, show_plot=True)
plot_energy_system_output_sorted(total_load, result_df_heatpump, result_df_discharge, show_plot=True)
plot_load_w_components(result_df_heatpump, result_df_discharge, total_load, show_plot=True)
plot_charge_discharge_process(result_df_charge, result_df_discharge, result_df_SOC, result_storage_capacity, show_plot=True)
plot_SOC(result_df_SOC, result_storage_capacity, show_plot=True)
