from funcs.paths import PARAMETERS_FILE
from funcs.plots import plot_temperatures, plot_prices, plot_gas_prices, \
                        plot_charge_discharge_process, plot_energy_system_output_sorted, \
                        plot_load_w_components, plot_SOC, plot_pv, plot_seasonal_storage, \
                        plot_network_losses, plot_energy_system_daily_stacked, \
                        plot_buffer_daily, plot_seasonal_daily, plot_pv_daily
from funcs.read_data import read_price_data, read_gas_price_data, read_pv_data, load_temperature_data
from funcs.era5_weather import load_era5_weather, compute_pv_generation, compute_cop, LAT as ERA5_LAT, LON as ERA5_LON
from funcs.energy_system_optimization import optimize_energy_system
from funcs.net_modelling import load_network_gpkg, build_graph, test_connectivity, create_pandapipes_network, \
                                export_res_pipe_gpkg, run_timeseries

import pandapipes
import pandas as pd
import warnings
import numpy as np


from funcs.read_data import read_parameters
parameters = read_parameters(PARAMETERS_FILE)


# -------------------------------------------------
# Netzsimulation
# -------------------------------------------------

# Trassierung importieren
gdf = load_network_gpkg(
    path="src/ACES-2026/Data/Trassierung_Jerrishoe.gpkg",
    layer="Trassierung_Jerrishoe",
)

# Netz bauen
graph = build_graph(gdf)
test_connectivity(graph, export_path="src/ACES-2026/Data/graph_komponenten.gpkg")

net, pipe_geoms, pipe_pairs = create_pandapipes_network(graph)

# Testrechnung
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*pressure is negative.*", category=UserWarning)
    pandapipes.pipeflow(net, mode="sequential")
export_res_pipe_gpkg(net, pipe_geoms, pipe_pairs, path="src/ACES-2026/Data/res_pipe_example.gpkg")

# Gebäudedaten laden und auf Trasse filtern
buildings_df = pd.read_csv(r"src/ACES-2026/Data/selected_267_profiles_2019_wide.csv")

# IDs aus GeoPackage (ID > 0 = echte Hausanschlüsse)
trasse_ids = set(gdf.loc[gdf["ID"] > 0, "ID"].astype(int).astype(str))
verfuegbar  = set(buildings_df.columns) - {"Datum"}
in_trasse   = sorted(trasse_ids & verfuegbar, key=lambda x: int(x))
nicht_in_df = trasse_ids - verfuegbar
buildings_df = buildings_df[["Datum"] + in_trasse]
print(f"Gebäude nach Trassierung: {len(in_trasse)} behalten "
      f"({len(verfuegbar) - len(in_trasse)} entfernt, "
      f"{len(nicht_in_df)} in Trasse aber nicht in CSV)")
# print(f'Gebäude-Dataframe: {buildings_df}')

# Zeitreihensimulation Netz
result_df = run_timeseries(net, buildings_df)

# Einzelne nicht konvergierte Zeitschritte (siehe pandapipes-Warnungen) linear interpolieren,
# damit kein NaN in die Optimierung durchschlägt (Python-sum() überspringt NaN nicht wie pandas)
nan_cols = ['mdot_kg_per_s', 't_supply_k', 't_return_k']
n_nan = result_df[nan_cols].isna().any(axis=1).sum()
if n_nan:
    print(f"Hinweis: {n_nan} nicht konvergierte Zeitschritte werden linear interpoliert.")
    result_df[nan_cols] = result_df[nan_cols].interpolate(limit_direction='both')

result_df.to_csv("src/ACES-2026/Data/result_timeseries.csv", index=False)
# print(f'Ergebnis-Dataframe (Netzsimulation): {result_df}')

# Spitzenlast filtern
peak_idx  = result_df['mdot_kg_per_s'].idxmax()
peak_mass_flow = result_df.loc[peak_idx, 'mdot_kg_per_s']
peak_date = result_df.loc[peak_idx, 'Datum']
peak_load_kW = peak_mass_flow * parameters['net_parameters']['cp'] * parameters['net_parameters']['delta_T'] / 1000
print(f"Spitzenlast: {peak_load_kW:.1f} kW  |  Massenstrom: {peak_mass_flow:.4f} kg/s  am  {peak_date}")

# Gesamtwärme der Pumpe: mdot × cp × tatsächliches ΔT (VL − RL an der Pumpe)
# Nicht Design-ΔT verwenden — sonst sind Rohrverluste unsichtbar!
cp = parameters['net_parameters']['cp']
result_df['delta_T_ist'] = result_df['t_supply_k'] - result_df['t_return_k']
result_df['load_kW'] = result_df['mdot_kg_per_s'] * cp * result_df['delta_T_ist'] / 1000
# print(result_df)

# Netzverluste berechnen
building_cols = [c for c in buildings_df.columns if c != 'Datum']
result_df['consumer_load_kW'] = buildings_df[building_cols].sum(axis=1).values
result_df['net_loss_kW'] = result_df['load_kW'] - result_df['consumer_load_kW']

jahresverbrauch_MWh    = result_df['load_kW'].sum() / 1000
gebaeude_MWh           = result_df['consumer_load_kW'].sum() / 1000
netzverlust_MWh        = result_df['net_loss_kW'].sum() / 1000
netzverlust_anteil_pct = netzverlust_MWh / jahresverbrauch_MWh * 100

print(f"\n--- Netzverluste ---")
print(f"Jahresgesamtverbrauch (Netz):  {jahresverbrauch_MWh:,.1f} MWh/a")
print(f"Gebäudeverbrauch (Summe):      {gebaeude_MWh:,.1f} MWh/a")
print(f"Netzverluste:                  {netzverlust_MWh:,.1f} MWh/a  ({netzverlust_anteil_pct:.1f} %)")

# Für die Rohrdimensionierung: Spitzenlastreihe simulieren
peak_row = buildings_df.iloc[[peak_idx]]  # doppelte Klammer → DataFrame statt Series
# print(f'Spitzenlast: {peak_row}')
peak_result_df = run_timeseries(net, peak_row)
export_res_pipe_gpkg(net, pipe_geoms, pipe_pairs, path="src/ACES-2026/Data/res_pipe_peak.gpkg")

# Dauerlinie in MW (Optimierung erwartet MW)
load = result_df.set_index('Datum')['load_kW'] / 1000

"""
# Angabe über mögliche Abwärmequellen (z.B. Industrie, Rechenzentren) OPTIONAL FÜR BERICHT
max_waste_heat_capacity = 0 # MW
waste_heat_cost = 0 # Euro/MWh

# Kann PV und Saisonalspeicher gebaut werden?
usable_area = 0 # m2
"""


# --------------------------------------------------
# Laden der Temperaturdaten (nicht nötig, evtl für COP der WP später)
# --------------------------------------------------

temperature = load_temperature_data(year=2019, lat=54.78, lon=9.43)

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
# Laden der Gaspreise
# --------------------------------------------------

gas_price = read_gas_price_data(
    path="src/ACES-2026/Data/",
    filename="Historic_THE_DA_Pegas.xlsx",
    load_data=ref_2024
)


# --------------------------------------------------
# Laden der PV-Daten (ERA5 + eigenes PV-Modell, Standort Jerrishoe, statt renewables.ninja)
# --------------------------------------------------
# ERA5 läuft nativ auf dem 2019-Kalender (wie `load`), daher direkt auf load.index
# ausrichten -- keine Schalttag-/Wochentag-Anpassung nötig (das war nur für die auf
# den 2024er-Kalender datierten renewables.ninja-Daten erforderlich, siehe unten).

weather_era5 = load_era5_weather(2019, lat=ERA5_LAT, lon=ERA5_LON)
# pv_capacity_MW=1.0 -> normiertes Kapazitätsfaktor-Profil (MW Ertrag pro MW installiert),
# NICHT auf initial_pv_capacity skalieren: das Optimierungsmodell multipliziert pv[t]
# selbst mit der von ihm dimensionierten m.pv_capacity-Variable (energy_system_optimization.py:310).
pv_era5 = compute_pv_generation(
    weather_era5, lat=ERA5_LAT, lon=ERA5_LON,
    surface_tilt=parameters['system_parameters']['PV']['surface_tilt'],
    surface_azimuth=parameters['system_parameters']['PV']['surface_azimuth'],
    pv_capacity_MW=1.0,
)
# load.index kommt aus der 'Datum'-Spalte der Gebäude-CSV (String, kein DatetimeIndex) ->
# für die zeitbasierte Ausrichtung/Interpolation separat in datetime konvertieren, ohne
# den Index von `load` selbst zu verändern (wird andernorts als String erwartet).
load_index_dt = pd.to_datetime(load.index)
pv = pv_era5.reindex(load_index_dt).interpolate(method='time').bfill().ffill().values

# --- Backup / alte Quelle (renewables.ninja), bei Bedarf reaktivierbar: ---
# pv = read_pv_data(
#     path="src/ACES-2026/Data/",
#     filename="ninja_pv_54.7833_9.4333_corrected.csv",
#     load_data=ref_2024
# )
# pv = pv[feb29]   # siehe Feb29-Filter unten


# --------------------------------------------------
# Temperaturabhängiger COP der Wärmepumpe (ERA5 T_amb_C + Carnot-Ansatz,
# statt statischem COP aus parameters.yaml)
# --------------------------------------------------
# Gleiche Ausrichtung wie bei pv_era5: ERA5 läuft nativ auf 2019, daher direkt
# auf load_index_dt reindizieren.

cop_era5 = compute_cop(weather_era5["T_amb_C"])
cop = cop_era5.reindex(load_index_dt).interpolate(method='time').bfill().ffill().values

# --- Backup / alter statischer COP, bei Bedarf reaktivierbar: ---
# cop = None   # optimize_energy_system()/calculate_lcoh() fallen dann auf den
#              # statischen COP aus parameters.yaml (system_parameters.HP.COP) zurück


# --------------------------------------------------
# Schalttag entfernen + Wochenprofil synchronisieren (Strompreise/Gaspreise)
# --------------------------------------------------
# Strom-/Gaspreise liegen auf 2024-Achse (8784 h, startet Mo).
# Load ist 2019-Netzberechnung (8760 h, startet Di).
#   1. Feb 29 entfernen → 8760 h
#   2. Um 24 h rotieren: Mo → Di = passt zu 2019-Wochenprofil

feb29 = ~((ref_2024.index.month == 2) & (ref_2024.index.day == 29))

electricity_price = electricity_price[feb29]
gas_price         = gas_price[feb29]

# Ersten Tag (Mo, 1.1.2024, 24 h) ans Ende → beide Preisreihen starten Di (= 2019)
electricity_price = np.concatenate([electricity_price[24:], electricity_price[:24]])
gas_price         = np.concatenate([gas_price[24:],         gas_price[:24]])


# --------------------------------------------------
# Energiesystemoptimierung
# --------------------------------------------------

results, result_df_heatpump, result_df_gas_boiler, result_df_charge, result_df_discharge, \
    result_df_SOC, result_storage_capacity, result_gas_boiler_capacity, result_hp_capacity, result_pv, \
    result_pv_feed_in, result_pv_capacity, result_seasonal_charge, result_seasonal_discharge, \
    result_seasonal_soc, result_seasonal_capacity \
    = optimize_energy_system(
        load, electricity_price, gas_price, pv,
        cop=cop,
        elec_price_mode="spot",       # "spot" | "tariff" | "hedge"
        elec_hedge_share=0.0,         # Anteil Festpreis bei mode="hedge" (0–1)
        gas_price_mode="tariff",        # "spot" | "tariff"
    )


# --------------------------------------------------
# Energiebilanz nach Komponente
# --------------------------------------------------

total_demand_MWh = load.sum()

wp_MWh        = result_df_heatpump.sum()
gas_MWh       = result_df_gas_boiler.sum()
buf_dis_MWh   = result_df_discharge.sum()
sea_dis_MWh   = result_seasonal_discharge.sum()
buf_ch_MWh    = result_df_charge.sum()
sea_ch_MWh    = result_seasonal_charge.sum()
pv_self_MWh   = (result_pv - result_pv_feed_in).sum()

from funcs.energy_system_optimization import storage_volume_to_MWh
buf_cap_MWh = storage_volume_to_MWh(result_storage_capacity)
sea_cap_MWh = storage_volume_to_MWh(result_seasonal_capacity)

print(f"\n{'='*58}")
print(f"  Jahreswärmebedarf (Netz):       {total_demand_MWh:>8.1f} MWh/a  (100 %)")
print(f"{'='*58}")
print(f"  Wärmepumpe:                     {wp_MWh:>8.1f} MWh/a  ({wp_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"  Gaskessel:                      {gas_MWh:>8.1f} MWh/a  ({gas_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"  Pufferspeicher Entladung:       {buf_dis_MWh:>8.1f} MWh/a  ({buf_dis_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"  Saisonalspeicher Entladung:     {sea_dis_MWh:>8.1f} MWh/a  ({sea_dis_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"{'='*58}")
print(f"  PV-Eigenverbrauch (→ WP):       {pv_self_MWh:>8.1f} MWh/a")
print(f"  Pufferspeicher Ladung:          {buf_ch_MWh:>8.1f} MWh/a")
print(f"  Saisonalspeicher Ladung:        {sea_ch_MWh:>8.1f} MWh/a")
print(f"{'='*58}")
print(f"  Wärmepumpe Kapazität:           {result_hp_capacity:>8.3f} MW")
print(f"  Gaskessel Kapazität:            {result_gas_boiler_capacity:>8.3f} MW")
print(f"  PV Kapazität:                   {result_pv_capacity:>8.3f} MW")
print(f"  Pufferspeicher Kapazität:       {buf_cap_MWh:>8.3f} MWh  ({result_storage_capacity:.3f} m³)")
print(f"  Saisonalspeicher Kapazität:     {sea_cap_MWh:>8.3f} MWh  ({result_seasonal_capacity:.3f} m³)")
print(f"{'='*58}")
buf_cycles = buf_ch_MWh / buf_cap_MWh if buf_cap_MWh > 0 else 0
sea_cycles = sea_ch_MWh / sea_cap_MWh if sea_cap_MWh > 0 else 0
print(f"  Pufferspeicher Zyklen/a:        {buf_cycles:>8.1f}  (Ladung / Kapazität)")
print(f"  Saisonalspeicher Zyklen/a:      {sea_cycles:>8.1f}  (Ladung / Kapazität)")
print(f"{'='*58}\n")

# --------------------------------------------------
# Pruefung der Netzentgeltstufe (Jahresbenutzungsdauer)
# --------------------------------------------------
# Das Netzentgelt der SH Netz kennt zwei Stufen (< / >= 2.500 Bh). Welche gilt,
# ergibt sich erst aus dem optimierten Dispatch, waehrend die Optimierung den
# Arbeitspreis schon vorher braucht. Die Annahme in parameters.yaml
# (price_parameters.electricity.vbh_class) wird hier ex post geprueft.

from funcs.read_data import read_parameters
_par = read_parameters()
_el = _par["price_parameters"]["electricity"]
_vbh_assumed = _el.get("vbh_class", "lower_2500VBH")

# Strombezug aus dem Netz rekonstruieren (Strombilanz der Optimierung)
_cop_arr = np.asarray(getattr(cop, "values", cop), dtype=float)
_P_grid = np.clip(
    np.asarray(result_df_heatpump) / _cop_arr
    + np.asarray(result_pv_feed_in)
    - np.asarray(result_pv),
    0.0, None,
)

_W_a = float(_P_grid.sum())      # MWh/a
_P_max = float(_P_grid.max())    # MW
_vbh = _W_a / _P_max if _P_max > 0 else 0.0
_vbh_actual = "higher_2500VBH" if _vbh >= 2500 else "lower_2500VBH"

print(f"{'='*58}")
print(f"  Netzbezug (Jahresarbeit):       {_W_a:>8.1f} MWh/a")
print(f"  Netzbezug (Spitzenlast):        {_P_max:>8.3f} MW")
print(f"  Jahresbenutzungsdauer:          {_vbh:>8.0f} h/a")
print(f"  Angenommene Stufe:              {_vbh_assumed}")
print(f"  Tatsaechliche Stufe:            {_vbh_actual}")
if _vbh_actual != _vbh_assumed:
    print("  >>> WARNUNG: Stufe passt nicht zur Annahme in parameters.yaml.")
    print(f"  >>> vbh_class auf '{_vbh_actual}' setzen und erneut rechnen.")
else:
    print("  Stufe konsistent.")
print(f"{'='*58}\n")

# --------------------------------------------------
# Plotting
# --------------------------------------------------

plot_network_losses(result_df, show_plot=True)
plot_prices(electricity_price, show_plot=True)
plot_gas_prices(gas_price, show_plot=False)
plot_temperatures(temperature, station_id="Flensburg", show_plot=False)

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

plot_charge_discharge_process(result_df_charge, 
                              result_df_discharge, 
                              result_df_SOC, 
                              result_storage_capacity, 
                              show_plot=True)

plot_SOC(result_df_SOC, result_storage_capacity, show_plot=True)
plot_pv(result_pv, result_pv_feed_in, result_pv_capacity, show_plot=True)

plot_seasonal_storage(result_seasonal_charge,
                      result_seasonal_discharge,
                      result_seasonal_soc,
                      result_seasonal_capacity,
                      show_plot=True)

plot_energy_system_daily_stacked(load,
                                 result_df_heatpump,
                                 result_df_gas_boiler,
                                 result_df_discharge,
                                 result_df_charge,
                                 result_seasonal_discharge,
                                 result_seasonal_charge,
                                 show_plot=True)

plot_buffer_daily(result_df_charge, result_df_discharge, show_plot=True)

plot_seasonal_daily(result_seasonal_charge, result_seasonal_discharge, show_plot=True)

plot_pv_daily(result_pv, result_pv_feed_in, result_pv_capacity, show_plot=True)

# --------------------------------------------------
# LCOH berechnen und plotten
# --------------------------------------------------
from funcs.LCOH import calculate_lcoh, plot_lcoh_pie

network_length = gdf["Length"].sum()   # [m]

lcoh, components = calculate_lcoh(
    demand=load, electricity_price=electricity_price, gas_price=gas_price,
    Q_hp=result_df_heatpump, charge=result_df_charge, discharge=result_df_discharge,
    Q_gas_boiler=result_df_gas_boiler, pv_availability=result_pv, pv_feed_in=result_pv_feed_in,
    storage_capacity_m3=result_storage_capacity, gas_boiler_capacity=result_gas_boiler_capacity,
    pv_capacity=result_pv_capacity, seasonal_capacity_m3=result_seasonal_capacity,
    network_length=network_length,
    hp_capacity=result_hp_capacity,   # direkt aus Optimierer, kein Rekonstruieren
    cop=cop,                          # gleicher COP-Verlauf wie in der Optimierung
    elec_price_mode="spot",           # gleicher Modus wie in der Optimierung
    gas_price_mode="tariff",          # gleicher Modus wie in der Optimierung
)

plot_lcoh_pie(components, lcoh, show_plot=True)