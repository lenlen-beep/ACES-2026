
from funcs.plots import plot_temperatures, plot_prices, plot_gas_prices, \
                        plot_charge_discharge_process, plot_energy_system_output_sorted, \
                        plot_load_w_components, plot_SOC, plot_pv, plot_seasonal_storage, \
                        plot_network_losses, plot_energy_system_daily_stacked, \
                        plot_buffer_daily, plot_seasonal_daily, plot_pv_daily, \
                        plot_gas_boiler
from funcs.read_data import read_price_data, read_gas_price_data, load_temperature_data
from funcs.era5_weather import load_era5_weather, compute_pv_generation, compute_cop, LAT as ERA5_LAT, LON as ERA5_LON
from funcs.energy_system_optimization import optimize_energy_system
from funcs.net_modelling import load_network_gpkg, build_graph, test_connectivity, create_pandapipes_network, \
                                export_res_pipe_gpkg, run_timeseries

import pandapipes
import pandas as pd
import warnings
import numpy as np


from funcs.read_data import read_parameters
parameters = read_parameters("src/ACES-2026/parameters.yaml")


# -------------------------------------------------
# Network simulation
# -------------------------------------------------

# Import route
gdf = load_network_gpkg(
    path="src/ACES-2026/Data/Trassierung_Jerrishoe_100pAQ.gpkg",
    layer="Trassierung_Jerrishoe",
)

# Build network
graph = build_graph(gdf)
test_connectivity(graph, export_path="src/ACES-2026/Data/graph_komponenten.gpkg")

net, pipe_geoms, pipe_pairs = create_pandapipes_network(graph)

# Load building data and filter to route
buildings_df = pd.read_csv(r"src/ACES-2026/Data/selected_267_profiles_2019_wide.csv")

trasse_ids = set(gdf.loc[gdf["ID"] > 0, "ID"].astype(int).astype(str))
verfuegbar  = set(buildings_df.columns) - {"Datum"}
in_trasse   = sorted(trasse_ids & verfuegbar, key=lambda x: int(x))
nicht_in_df = trasse_ids - verfuegbar
buildings_df = buildings_df[["Datum"] + in_trasse]
print(f"Buildings after route filter: {len(in_trasse)} kept "
      f"({len(verfuegbar) - len(in_trasse)} removed, "
      f"{len(nicht_in_df)} in route but not in CSV)")

# Determine peak load directly from building profiles
building_cols = [c for c in buildings_df.columns if c != 'Datum']
peak_idx  = buildings_df[building_cols].sum(axis=1).idxmax()
peak_date = buildings_df.loc[peak_idx, 'Datum']
peak_load_kW = buildings_df.loc[peak_idx, building_cols].sum()
print(f"Peak load (buildings): {peak_load_kW:.1f} kW  at  {peak_date}")

# Pipe dimensioning: simulate single peak-load timestep → dimension pipes
from funcs.net_modelling import fix_pipe_orientations, dimension_pipes
peak_row = buildings_df.iloc[[peak_idx]]
peak_result_df = run_timeseries(net, peak_row)
fix_pipe_orientations(net)
df_dimensionierung = dimension_pipes(net, parameters)
export_res_pipe_gpkg(net, pipe_geoms, pipe_pairs, path="src/ACES-2026/Data/res_pipe_peak.gpkg")

# Full time-series simulation with dimensioned pipes
result_df = run_timeseries(net, buildings_df)
result_df.to_csv("src/ACES-2026/Data/result_timeseries.csv", index=False)

# Total heat from pump: mdot × cp × actual ΔT (supply − return at pump)
cp = parameters['net_parameters']['cp']
result_df['delta_T_ist'] = result_df['t_supply_k'] - result_df['t_return_k']
result_df['load_kW'] = result_df['mdot_kg_per_s'] * cp * result_df['delta_T_ist'] / 1000

# Calculate network losses
result_df['consumer_load_kW'] = buildings_df[building_cols].sum(axis=1).values
result_df['net_loss_kW'] = result_df['load_kW'] - result_df['consumer_load_kW']

jahresverbrauch_MWh    = result_df['load_kW'].sum() / 1000
gebaeude_MWh           = result_df['consumer_load_kW'].sum() / 1000
netzverlust_MWh        = result_df['net_loss_kW'].sum() / 1000
netzverlust_anteil_pct = netzverlust_MWh / jahresverbrauch_MWh * 100

print(f"\n--- Network losses ---")
print(f"Annual total (network):        {jahresverbrauch_MWh:,.1f} MWh/a")
print(f"Building consumption (sum):    {gebaeude_MWh:,.1f} MWh/a")
print(f"Network losses:                {netzverlust_MWh:,.1f} MWh/a  ({netzverlust_anteil_pct:.1f} %)")

# Load duration curve in MW (optimiser expects MW)
load = result_df.set_index('Datum')['load_kW'] / 1000


# --------------------------------------------------
# Load temperature data (not required, possibly for heat pump COP later)
# --------------------------------------------------

temperature = load_temperature_data(year=2019, lat=54.78, lon=9.43)

# Reference index for 2024 data (prices, PV): leap year = 8784 h
ref_2024 = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=8784, freq="1h"))


# --------------------------------------------------
# Load and prepare electricity prices
# --------------------------------------------------

electricity_price = read_price_data(
    path="src/ACES-2026/Data/",
    filename="Gro_handelspreise_202401010000_202501010000_Stunde.xlsx",
    load_data=ref_2024
)


# --------------------------------------------------
# Load gas prices
# --------------------------------------------------

gas_price = read_gas_price_data(
    path="src/ACES-2026/Data/",
    filename="Historic_THE_DA_Pegas.xlsx",
    load_data=ref_2024
)


# --------------------------------------------------
# Load PV data (ERA5 + custom PV model, location Jerrishoe)
# --------------------------------------------------
# ERA5 runs natively on the 2019 calendar (like `load`), so align directly to
# load.index — no leap-day / weekday shift needed (that was only required for
# the renewables.ninja data dated on the 2024 calendar, see below).

weather_era5 = load_era5_weather(2019, lat=ERA5_LAT, lon=ERA5_LON)
# pv_capacity_MW=1.0 -> normalised capacity-factor profile (MW yield per MW installed);
# do NOT scale to initial_pv_capacity: the optimisation model multiplies pv[t]
# by its own m.pv_capacity variable (energy_system_optimization.py:310).
pv_era5 = compute_pv_generation(
    weather_era5, lat=ERA5_LAT, lon=ERA5_LON,
    surface_tilt=parameters['system_parameters']['PV']['surface_tilt'],
    surface_azimuth=parameters['system_parameters']['PV']['surface_azimuth'],
    pv_capacity_MW=1.0,
)
# load.index comes from the 'Datum' column of the building CSV (string, not DatetimeIndex) →
# convert to datetime separately for time-based alignment/interpolation, without
# modifying the index of `load` itself (expected as string elsewhere).
load_index_dt = pd.to_datetime(load.index)
pv = pv_era5.reindex(load_index_dt).interpolate(method='time').bfill().ffill().values


# --------------------------------------------------
# Temperature-dependent heat pump COP (ERA5 T_amb_C + Carnot approach,
# instead of static COP from parameters.yaml)
# --------------------------------------------------
# Same alignment as for pv_era5: ERA5 runs natively on 2019, so reindex
# directly to load_index_dt.

cop_era5 = compute_cop(weather_era5["T_amb_C"])
cop = cop_era5.reindex(load_index_dt).interpolate(method='time').bfill().ffill().values

# --- Backup / old static COP, re-enable if needed: ---
# cop = None   # optimize_energy_system()/calculate_lcoh() then fall back to the
#              # static COP from parameters.yaml (system_parameters.HP.COP)


# --------------------------------------------------
# Remove leap day + synchronise weekly profile (electricity/gas prices)
# --------------------------------------------------
# Electricity/gas prices are on the 2024 axis (8784 h, starts Monday).
# Load is from the 2019 network calculation (8760 h, starts Tuesday).
#   1. Remove Feb 29 → 8760 h
#   2. Rotate by 24 h: Monday → Tuesday = matches 2019 weekly profile

feb29 = ~((ref_2024.index.month == 2) & (ref_2024.index.day == 29))

electricity_price = electricity_price[feb29]
gas_price         = gas_price[feb29]

# Move first day (Mon, 01-01-2024, 24 h) to end → both price series start Tuesday (= 2019)
electricity_price = np.concatenate([electricity_price[24:], electricity_price[:24]])
gas_price         = np.concatenate([gas_price[24:],         gas_price[:24]])


# --------------------------------------------------
# Energy system optimisation
# --------------------------------------------------

results, result_df_heatpump, result_df_gas_boiler, result_df_charge, result_df_discharge, \
    result_df_SOC, result_storage_capacity, result_gas_boiler_capacity, result_hp_capacity, result_pv, \
    result_pv_feed_in, result_pv_capacity, result_seasonal_charge, result_seasonal_discharge, \
    result_seasonal_soc, result_seasonal_capacity \
    = optimize_energy_system(
        load, electricity_price, gas_price, pv,
        cop=cop,
        elec_price_mode="spot",       # "spot" | "tariff" | "hedge"
        elec_hedge_share=0.0,         # fixed-price share for mode="hedge" (0–1)
        gas_price_mode="tariff",        # "spot" | "tariff"
    )


# --------------------------------------------------
# Energy balance by component
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
print(f"  Annual heat demand (network):   {total_demand_MWh:>8.1f} MWh/a  (100 %)")
print(f"{'='*58}")
print(f"  Heat pump:                      {wp_MWh:>8.1f} MWh/a  ({wp_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"  Gas boiler:                     {gas_MWh:>8.1f} MWh/a  ({gas_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"  Buffer storage discharge:       {buf_dis_MWh:>8.1f} MWh/a  ({buf_dis_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"  Seasonal storage discharge:     {sea_dis_MWh:>8.1f} MWh/a  ({sea_dis_MWh/total_demand_MWh*100:>5.1f} %)")
print(f"{'='*58}")
print(f"  PV self-consumption (→ HP):     {pv_self_MWh:>8.1f} MWh/a")
print(f"  Buffer storage charging:        {buf_ch_MWh:>8.1f} MWh/a")
print(f"  Seasonal storage charging:      {sea_ch_MWh:>8.1f} MWh/a")
print(f"{'='*58}")
print(f"  Heat pump capacity:             {result_hp_capacity:>8.3f} MW")
print(f"  Gas boiler capacity:            {result_gas_boiler_capacity:>8.3f} MW")
print(f"  PV capacity:                    {result_pv_capacity:>8.3f} MW")
print(f"  Buffer storage capacity:        {buf_cap_MWh:>8.3f} MWh  ({result_storage_capacity:.3f} m³)")
print(f"  Seasonal storage capacity:      {sea_cap_MWh:>8.3f} MWh  ({result_seasonal_capacity:.3f} m³)")
print(f"{'='*58}")
buf_cycles = buf_ch_MWh / buf_cap_MWh if buf_cap_MWh > 0 else 0
sea_cycles = sea_ch_MWh / sea_cap_MWh if sea_cap_MWh > 0 else 0
print(f"  Buffer storage cycles/a:        {buf_cycles:>8.1f}  (charging / capacity)")
print(f"  Seasonal storage cycles/a:      {sea_cycles:>8.1f}  (charging / capacity)")
print(f"{'='*58}\n")

# --------------------------------------------------
# Check network tariff tier (annual utilisation hours)
# --------------------------------------------------
# The SH Netz network charge has two tiers (< / >= 2,500 h). Which applies
# depends on the optimised dispatch, while the optimiser needs the commodity
# charge beforehand. The assumption in parameters.yaml
# (price_parameters.electricity.vbh_class) is verified here ex post.

_par = parameters
_el = _par["price_parameters"]["electricity"]
_vbh_assumed = _el.get("vbh_class", "lower_2500VBH")

# Reconstruct grid electricity draw (optimisation electricity balance)
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
print(f"  Grid draw (annual energy):      {_W_a:>8.1f} MWh/a")
print(f"  Grid draw (peak load):          {_P_max:>8.3f} MW")
print(f"  Annual utilisation hours:       {_vbh:>8.0f} h/a")
print(f"  Assumed tier:                   {_vbh_assumed}")
print(f"  Actual tier:                    {_vbh_actual}")
if _vbh_actual != _vbh_assumed:
    print("  >>> WARNING: tier does not match assumption in parameters.yaml.")
    print(f"  >>> Set vbh_class to '{_vbh_actual}' and re-run.")
else:
    print("  Tier consistent.")
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

plot_gas_boiler(result_df_gas_boiler, result_gas_boiler_capacity, show_plot=True)

# --------------------------------------------------
# Calculate and plot LCOH
# --------------------------------------------------
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from funcs.LCOH import calculate_lcoh, plot_lcoh_pie

network_length = gdf["Length_m"].sum()   # [m]

lcoh, components = calculate_lcoh(
    demand=load, electricity_price=electricity_price, gas_price=gas_price,
    Q_hp=result_df_heatpump, charge=result_df_charge, discharge=result_df_discharge,
    Q_gas_boiler=result_df_gas_boiler, pv_availability=result_pv, pv_feed_in=result_pv_feed_in,
    storage_capacity_m3=result_storage_capacity, gas_boiler_capacity=result_gas_boiler_capacity,
    pv_capacity=result_pv_capacity, seasonal_capacity_m3=result_seasonal_capacity,
    network_length=network_length,
    hp_capacity=result_hp_capacity,   # directly from optimiser, no reconstruction needed
    cop=cop,                          # same COP profile as used in optimisation
    elec_price_mode="spot",           # same mode as in optimisation
    gas_price_mode="tariff",          # same mode as in optimisation
)

plot_lcoh_pie(components, lcoh, show_plot=True)
