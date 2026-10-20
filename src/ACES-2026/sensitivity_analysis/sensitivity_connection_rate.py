#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_connection_rate.py – Connection-rate sensitivity (approximation mode)

Step 0  (100 %): full Pandapipes simulation → produces hourly network-loss
                 profile as reference.
Steps 1…N:      Pandapipes skipped. Building consumption is summed directly
                from profiles; network losses scale proportionally with pipe
                length (net_loss ~ L_config / L_ref).
                The energy-system optimisation runs fully for every step.

Usage:
    cd <repo-root>
    python src/ACES-2026/sensitivity_analysis/sensitivity_connection_rate.py
"""

from __future__ import annotations
import os, sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec

# Parent directory (src/ACES-2026/) added so that `funcs.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from funcs.read_data       import read_parameters, read_price_data, read_gas_price_data
from funcs.era5_weather    import (load_era5_weather, compute_pv_generation, compute_cop,
                                   LAT as ERA5_LAT, LON as ERA5_LON)
from funcs.energy_system_optimization import optimize_energy_system, storage_volume_to_MWh
from funcs.LCOH            import calculate_lcoh
from funcs.net_modelling   import (load_network_gpkg, build_graph, create_pandapipes_network,
                                   run_timeseries, fix_pipe_orientations, dimension_pipes)
import pandapipes


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

GPKG_PATH     = "src/ACES-2026/Data/Trassierung_Jerrishoe_100pAQ.gpkg"
GPKG_LAYER    = "Trassierung_Jerrishoe"
BUILDINGS_CSV = "src/ACES-2026/Data/selected_267_profiles_2019_wide.csv"
DATA_PATH     = "src/ACES-2026/Data/"
RESULTS_CSV   = "src/ACES-2026/Data/sensitivity_results.csv"
PLOTS_DIR     = "src/ACES-2026/plots"
PARAMS_FILE   = "src/ACES-2026/parameters.yaml"

MIN_RATE   = 0.10   # lower connection rate
MAX_RATE   = 1.00   # upper connection rate (= all buildings in layer)
STEP_SIZE  = 4      # buildings per step (1 = finest resolution, 4 ≈ 3 h runtime)


# ─────────────────────────────────────────────────────────────────────────────
#  Layout and plot styles
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman', 'Times', 'DejaVu Serif']

LABEL_FS  = 20
TICK_FS   = 20
LEGEND_FS = 16
TITLE_FS  = 20

# Component colours (identical to plots.py / LCOH.py)
COLOR_WP       = "#00395B"
COLOR_GAS      = "#C17A2F"
COLOR_SPEICHER = "#769D7B"
COLOR_SAISONAL = "#2F6B4F"
COLOR_PV       = "#C8A84B"
COLOR_NETZ     = "#A0463A"
COLOR_PUMP     = "#888888"
COLOR_OM       = "#AAAAAA"
COLOR_NETZENT  = "#D4D4D4"
COLOR_LCOH     = "#8B2635"

# Cost components: key → (label, colour) — same grouping as pie chart
COST_GROUPS: dict[str, tuple[str, str]] = {
    "c_wp":    ("Heat pump",                  COLOR_WP),
    "c_gas":   ("Gas boiler",                 COLOR_GAS),
    "c_buf":   ("Buffer storage",             COLOR_SPEICHER),
    "c_sea":   ("Seasonal storage",           COLOR_SAISONAL),
    "c_grid":  ("District heating network",   COLOR_NETZ),
    "c_pump":  ("Pump electricity",           COLOR_PUMP),
    "c_om":    ("O&M (1.5 % CAPEX)",          COLOR_OM),
    "c_netz":  ("Grid capacity charge",       COLOR_NETZENT),
    "c_pv":    ("PV (net)",                   COLOR_PV),   # can be negative
}

# Component sizes: key → (label, colour)
COMP_SIZES: dict[str, tuple[str, str]] = {
    "hp_cap_MW":   ("Heat pump in MW",          COLOR_WP),
    "gas_cap_MW":  ("Gas boiler in MW",         COLOR_GAS),
    "buf_cap_MWh": ("Buffer storage in MWh",    COLOR_SPEICHER),
    "sea_cap_MWh": ("Seasonal storage in MWh",  COLOR_SAISONAL),
    "pv_cap_MW":   ("PV in MW",                 COLOR_PV),
}

# X-axes for component size plots
X_AXES: list[tuple[str, str]] = [
    ("annual_heat_MWh", "Annual heat demand in MWh"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Shared data (loaded once)
# ─────────────────────────────────────────────────────────────────────────────

parameters = read_parameters(PARAMS_FILE)

print("Loading ERA5 data ...")
weather_era5 = load_era5_weather(2019, lat=ERA5_LAT, lon=ERA5_LON)
pv_era5      = compute_pv_generation(
    weather_era5,
    lat=ERA5_LAT, lon=ERA5_LON,
    surface_tilt=parameters['system_parameters']['PV']['surface_tilt'],
    surface_azimuth=parameters['system_parameters']['PV']['surface_azimuth'],
    pv_capacity_MW=1.0,
)
cop_era5 = compute_cop(weather_era5["T_amb_C"])

print("Loading price data ...")
ref_2024 = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=8784, freq="1h"))
elec_raw = read_price_data(
    path=DATA_PATH,
    filename="Gro_handelspreise_202401010000_202501010000_Stunde.xlsx",
    load_data=ref_2024,
)
gas_raw = read_gas_price_data(
    path=DATA_PATH,
    filename="Historic_THE_DA_Pegas.xlsx",
    load_data=ref_2024,
)
_feb29    = ~((ref_2024.index.month == 2) & (ref_2024.index.day == 29))
elec_base = np.concatenate([elec_raw[_feb29][24:], elec_raw[_feb29][:24]])
gas_base  = np.concatenate([gas_raw[_feb29][24:],  gas_raw[_feb29][:24]])

print("Loading building profiles ...")
buildings_all = pd.read_csv(BUILDINGS_CSV)
buildings_all["Datum"] = pd.to_datetime(buildings_all["Datum"])

print(f"Loading GPKG '{GPKG_LAYER}' ...")
gdf_full = load_network_gpkg(path=GPKG_PATH, layer=GPKG_LAYER)

all_bldg_ids = sorted(
    gdf_full.loc[gdf_full["ID"] > 0, "ID"].astype(float).astype(int).tolist()
)
N_TOTAL = len(all_bldg_ids)
print(f"\n{N_TOTAL} buildings in '{GPKG_LAYER}'  (ID {all_bldg_ids[0]}–{all_bldg_ids[-1]})\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Removal order: alternating smallest / largest building ID
# ─────────────────────────────────────────────────────────────────────────────

def generate_removal_order(building_ids: list[int]) -> list[int]:
    ids = sorted(building_ids)
    order: list[int] = []
    l, r = 0, len(ids) - 1
    while l <= r:
        order.append(ids[l])
        if l != r:
            order.append(ids[r])
        l += 1; r -= 1
    return order


removal_order = generate_removal_order(all_bldg_ids)
n_to_remove   = int(np.floor(N_TOTAL * (MAX_RATE - MIN_RATE)))
print(f"Run {MAX_RATE*100:.0f} % → {MIN_RATE*100:.0f} %:  "
      f"{n_to_remove} buildings will be removed → {n_to_remove + 1} steps\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: extract cost components from components dict
# ─────────────────────────────────────────────────────────────────────────────

def _extract_costs(components: dict) -> dict:
    """Groups LCOH components into 9 categories (same as pie chart). Values in €/MWh."""
    g = components
    pv_capex   = g["PV (CAPEX)"]["eur_per_mwh"]
    pv_revenue = g["PV-Einspeiseerlös"]["eur_per_mwh"]   # already negative

    return {
        "c_wp":   g["Wärmepumpe (CAPEX)"]["eur_per_mwh"] + g["Strombezug WP (OPEX)"]["eur_per_mwh"],
        "c_gas":  g["Gaskessel (CAPEX)"]["eur_per_mwh"]  + g["Gas-Brennstoff (OPEX)"]["eur_per_mwh"],
        "c_buf":  g["Pufferspeicher (CAPEX)"]["eur_per_mwh"],
        "c_sea":  g["Saisonalspeicher (CAPEX)"]["eur_per_mwh"],
        "c_grid": g["Wärmenetz (CAPEX)"]["eur_per_mwh"],
        "c_pump": g["Pumpstrom (OPEX)"]["eur_per_mwh"],
        "c_om":   g["O&M (1,5 % CAPEX)"]["eur_per_mwh"],
        "c_netz": g["Netzentgelt Leistungspreis"]["eur_per_mwh"],
        "c_pv":   pv_capex + pv_revenue,   # net (can be negative)
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Optimisation + LCOH (identical for all steps)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_and_lcoh(load: pd.Series, cop: np.ndarray, pv: np.ndarray,
                      network_length_m: float, label: str) -> dict:
    """Optimisation + LCOH. Returns partial result dict including cost components."""

    print(f"  [{label}] Optimising ...", end=" ", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        (_, df_hp, df_gas, df_charge, df_discharge,
         _soc, cap_buf, cap_gas, hp_cap, pv_res, pv_feed,
         cap_pv, _sea_ch, _sea_dis, _sea_soc, cap_sea
         ) = optimize_energy_system(
            load, elec_base, gas_base, pv,
            cop=cop,
            elec_price_mode="spot",
            gas_price_mode="tariff",
        )
    print("OK.", flush=True)

    lcoh, components = calculate_lcoh(
        demand=load,
        electricity_price=elec_base,
        gas_price=gas_base,
        Q_hp=df_hp,
        charge=df_charge,
        discharge=df_discharge,
        Q_gas_boiler=df_gas,
        pv_availability=pv_res,
        pv_feed_in=pv_feed,
        storage_capacity_m3=cap_buf,
        gas_boiler_capacity=cap_gas,
        pv_capacity=cap_pv,
        seasonal_capacity_m3=cap_sea,
        network_length=network_length_m,
        hp_capacity=hp_cap,
        cop=cop,
        elec_price_mode="spot",
        gas_price_mode="tariff",
    )

    costs = _extract_costs(components)

    return {
        "lcoh":        lcoh,
        "hp_cap_MW":   float(hp_cap),
        "gas_cap_MW":  float(cap_gas),
        "buf_cap_MWh": storage_volume_to_MWh(cap_buf),
        "sea_cap_MWh": storage_volume_to_MWh(cap_sea),
        "pv_cap_MW":   float(cap_pv),
        **costs,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Step 0: full calculation (Pandapipes), create reference loss profile
# ─────────────────────────────────────────────────────────────────────────────

def run_step0_full(active_ids: list[int]) -> tuple[dict, np.ndarray, float, pd.Index]:
    label = "100.0%"
    t0    = time.perf_counter()

    active_set = set(active_ids)
    _id_int    = gdf_full["ID"].fillna(0).astype(float).astype(int)
    gdf        = gdf_full[(_id_int == 0) | (_id_int.isin(active_set))].copy()

    graph = build_graph(gdf)
    net, _geoms, _pairs = create_pandapipes_network(graph)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pandapipes.pipeflow(net, mode="sequential")

    avail   = set(buildings_all.columns) - {"Datum"}
    ids_str = sorted({str(i) for i in active_ids} & avail, key=int)
    n_hx    = len(net.heat_exchanger)
    if len(ids_str) != n_hx:
        print(f"  [{label}] Info: {len(ids_str)} profiles vs. {n_hx} HX – using first {n_hx}")
        ids_str = ids_str[:n_hx]
    bdf   = buildings_all[["Datum"] + ids_str].copy()
    bcols = [c for c in bdf.columns if c != 'Datum']

    # Peak load from building profiles → pipe dimensioning before annual simulation
    peak_idx = bdf[bcols].sum(axis=1).idxmax()
    peak_row = bdf.iloc[[peak_idx]]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        run_timeseries(net, peak_row)
    fix_pipe_orientations(net)
    dimension_pipes(net, parameters)

    print(f"  [{label}] Annual simulation ...", end=" ", flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        result_df = run_timeseries(net, bdf)
    print("OK.", flush=True)

    cp = parameters['net_parameters']['cp']
    result_df['delta_T_ist']      = result_df['t_supply_k'] - result_df['t_return_k']
    result_df['load_kW']          = result_df['mdot_kg_per_s'] * cp * result_df['delta_T_ist'] / 1000
    result_df['consumer_load_kW'] = bdf[bcols].sum(axis=1).values
    result_df['net_loss_kW']      = result_df['load_kW'] - result_df['consumer_load_kW']

    load        = result_df.set_index('Datum')['load_kW'] / 1000
    datum_index = load.index
    load_idx_dt = pd.to_datetime(datum_index)
    pv  = pv_era5.reindex(load_idx_dt).interpolate(method='time').bfill().ffill().values
    cop = cop_era5.reindex(load_idx_dt).interpolate(method='time').bfill().ffill().values

    network_length_m = gdf["Length_m"].sum()
    opt_res = optimize_and_lcoh(load, cop, pv, network_length_m, label)

    annual_kwh = float(load.sum()) * 1000
    peak_kW    = float(result_df['load_kW'].max())

    print(f"  [{label}] {(time.perf_counter()-t0)/60:.1f} min  |  "
          f"LCOH = {opt_res['lcoh']:.1f} €/MWh  |  "
          f"HLD = {annual_kwh/network_length_m:.0f} kWh/m")

    result = {
        "n_buildings":      len(active_ids),
        "annual_heat_kwh":  annual_kwh,
        "peak_load_kW":     peak_kW,
        "wld_kwh_m":        annual_kwh / network_length_m,
        "network_length_m": network_length_m,
        **opt_res,
    }
    return result, result_df['net_loss_kW'].values.copy(), network_length_m, datum_index


# ─────────────────────────────────────────────────────────────────────────────
#  Steps 1…N: approximation (without Pandapipes annual simulation)
# ─────────────────────────────────────────────────────────────────────────────

def run_step_approx(active_ids: list[int], net_loss_kw_ref: np.ndarray,
                    length_ref_m: float, datum_index: pd.Index,
                    rate_pct: float) -> dict:
    """
    Fast approximation: consumption summed from profiles,
    network losses scaled proportionally to pipe length.
    """
    label = f"{rate_pct:.1f}%"
    t0    = time.perf_counter()

    active_set = set(active_ids)
    _id_int    = gdf_full["ID"].fillna(0).astype(float).astype(int)
    gdf        = gdf_full[(_id_int == 0) | (_id_int.isin(active_set))].copy()
    network_length_m = gdf["Length_m"].sum()

    avail   = set(buildings_all.columns) - {"Datum"}
    ids_str = sorted({str(i) for i in active_ids} & avail, key=int)
    bcols   = ids_str
    consumer_load_kw = buildings_all[bcols].sum(axis=1).values   # [kW]

    # Scale network losses (heat loss ∝ pipe length)
    heat_loss_kw = net_loss_kw_ref * (network_length_m / length_ref_m)
    load_kw_arr  = consumer_load_kw + heat_loss_kw
    load         = pd.Series(load_kw_arr / 1000, index=datum_index)   # [MW]

    load_idx_dt = pd.to_datetime(datum_index)
    pv  = pv_era5.reindex(load_idx_dt).interpolate(method='time').bfill().ffill().values
    cop = cop_era5.reindex(load_idx_dt).interpolate(method='time').bfill().ffill().values

    opt_res = optimize_and_lcoh(load, cop, pv, network_length_m, label)

    annual_kwh = float(load.sum()) * 1000
    peak_kW    = float(load_kw_arr.max())

    print(f"  [{label}] {(time.perf_counter()-t0)/60:.1f} min  |  "
          f"LCOH = {opt_res['lcoh']:.1f} €/MWh  |  "
          f"HLD = {annual_kwh/network_length_m:.0f} kWh/m")

    return {
        "n_buildings":      len(active_ids),
        "annual_heat_kwh":  annual_kwh,
        "peak_load_kW":     peak_kW,
        "wld_kwh_m":        annual_kwh / network_length_m,
        "network_length_m": network_length_m,
        **opt_res,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Sensitivity loop
# ─────────────────────────────────────────────────────────────────────────────

removed_ids:  list[int]  = []
results_list: list[dict] = []
net_loss_ref: np.ndarray | None = None
length_ref:   float | None = None
datum_idx:    pd.Index | None = None

# Test mode: only run specific connection rates (comment out for full run)
_TEST_RATES  = [1.00, 0.85, 0.70]
step_indices = sorted(set(int(round(N_TOTAL * (1.0 - r))) for r in _TEST_RATES))

# Full run (comment in, comment out test block above):
# step_indices = list(range(0, n_to_remove + 1, STEP_SIZE))
# if step_indices[-1] != n_to_remove:
#     step_indices.append(n_to_remove)

for loop_idx, step in enumerate(step_indices):
    n_active = N_TOTAL - step
    rate_pct = n_active / N_TOTAL * 100
    active   = [i for i in all_bldg_ids if i not in set(removal_order[:step])]

    print(f"\n{'─'*62}")
    print(f"  Step {loop_idx+1}/{len(step_indices)}  |  {n_active} buildings  ({rate_pct:.1f} % connection rate)")
    print(f"{'─'*62}")

    try:
        if step == 0:
            res, net_loss_ref, length_ref, datum_idx = run_step0_full(active)
        else:
            res = run_step_approx(active, net_loss_ref, length_ref, datum_idx, rate_pct)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        res = None

    if res is not None:
        res["connection_rate_pct"] = rate_pct
        res["step"] = step
        results_list.append(res)

results_df = pd.DataFrame(results_list)
results_df["annual_heat_MWh"] = results_df["annual_heat_kwh"] / 1000.0
os.makedirs(DATA_PATH, exist_ok=True)
results_df.to_csv(RESULTS_CSV, index=False)

print(f"\n{'='*62}")
print(f"  Results saved: {RESULTS_CSV}")
print(f"{'='*62}")
print(results_df[[
    "connection_rate_pct", "n_buildings",
    "annual_heat_kwh", "peak_load_kW", "lcoh", "wld_kwh_m",
]].to_string(index=False, float_format="{:,.1f}".format))


# ─────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ax_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.tick_params(labelsize=TICK_FS)


def _save_fig(fig, filename: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  → {path}")


print("\nCreating plots ...")

x_rate = results_df["connection_rate_pct"].values


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1: LCOH vs. connection rate
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
ax.plot(x_rate, results_df["lcoh"].values,
        color=COLOR_LCOH, lw=2.5, ms=9, zorder=3)
ax.set_xlabel("Connection rate in %", fontsize=LABEL_FS)
ax.set_ylabel("LCOH in €/MWh", fontsize=LABEL_FS)
_ax_style(ax)
fig.tight_layout()
_save_fig(fig, "sensitivity_lcoh_connection_rate.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2: LCOH vs. heat line density
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
ax.plot(results_df["wld_kwh_m"].values, results_df["lcoh"].values,
        color=COLOR_GAS, lw=2.5, ms=9, zorder=3)
ax.set_xlabel("Heat line density in kWh/m", fontsize=LABEL_FS)
ax.set_ylabel("LCOH in €/MWh", fontsize=LABEL_FS)
_ax_style(ax)
fig.tight_layout()
_save_fig(fig, "sensitivity_lcoh_waermeliniendichte.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3: Stacked area – cost components vs. connection rate
# ─────────────────────────────────────────────────────────────────────────────

# Sort ascending so stackplot produces a continuous area
_df = results_df.sort_values("connection_rate_pct").reset_index(drop=True)
_x  = _df["connection_rate_pct"].values

# Positive cost groups (stack order from bottom)
_stack_keys = ["c_wp", "c_gas", "c_buf", "c_sea", "c_grid", "c_pump", "c_om", "c_netz"]

_pv_vals = _df["c_pv"].values
_pv_pos  = np.maximum(_pv_vals, 0)
_pv_neg  = np.minimum(_pv_vals, 0)

_stack_data   = [_df[k].values for k in _stack_keys] + [_pv_pos]
_stack_colors = [COST_GROUPS[k][1] for k in _stack_keys] + [COLOR_PV]
_stack_labels = [COST_GROUPS[k][0] for k in _stack_keys] + [COST_GROUPS["c_pv"][0]]

fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")

ax.stackplot(_x, *_stack_data,
             labels=_stack_labels, colors=_stack_colors, alpha=0.85, zorder=2)

# PV feed-in credit: negative part as transparent area below zero line
if np.any(_pv_neg < 0):
    ax.fill_between(_x, _pv_neg, 0,
                    color=COLOR_PV, alpha=0.35, zorder=2, label=None)

# Total LCOH line as reference
ax.plot(_x, _df["lcoh"].values,
        color="#1A1A1A", lw=2, ls="--", zorder=4, label="LCOH total")

ax.axhline(0, color="#555555", lw=0.7, zorder=3)

ax.set_xlabel("Connection rate in %", fontsize=LABEL_FS)
ax.set_ylabel("LCOH components in €/MWh", fontsize=LABEL_FS)

# Legend outside right
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=LEGEND_FS, frameon=False,
          loc="upper left", bbox_to_anchor=(1.01, 1.0))

_ax_style(ax)
fig.tight_layout()
_save_fig(fig, "sensitivity_cost_breakdown.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4: Component sizes – bundled (max. 3 per row)
# ─────────────────────────────────────────────────────────────────────────────

_MAX_COLS    = 3
_comp_list   = list(COMP_SIZES.items())
_N_COMP      = len(_comp_list)
_N_COMP_ROWS = (_N_COMP + _MAX_COLS - 1) // _MAX_COLS   # ceil(5/3) = 2
_n_grid_rows = len(X_AXES) * _N_COMP_ROWS
_GRID_SUBCOLS = _MAX_COLS * 2  # 6 subdivisions → incomplete rows centre cleanly

fig = plt.figure(figsize=(6 * _MAX_COLS, 5 * _n_grid_rows), facecolor="white")
gs  = mgridspec.GridSpec(_n_grid_rows, _GRID_SUBCOLS, figure=fig)

for x_idx, (x_col, x_label) in enumerate(X_AXES):
    x_vals = results_df[x_col].values
    for sub_row_idx in range(_N_COMP_ROWS):
        start_comp   = sub_row_idx * _MAX_COLS
        end_comp     = min(start_comp + _MAX_COLS, _N_COMP)
        n_in_row     = end_comp - start_comp
        _span        = _GRID_SUBCOLS // _MAX_COLS                        # 2
        _col_offset  = (_GRID_SUBCOLS - n_in_row * _span) // 2          # 0 if full, 1 if 2 items
        global_row   = x_idx * _N_COMP_ROWS + sub_row_idx

        for i in range(n_in_row):
            comp_idx = start_comp + i
            comp_key, (comp_ylabel, comp_color) = _comp_list[comp_idx]
            col_start = _col_offset + i * _span
            col_end   = col_start + _span
            ax = fig.add_subplot(gs[global_row, col_start:col_end])
            ax.plot(x_vals, results_df[comp_key].values,
                    color=comp_color, lw=2.5, ms=8, zorder=3)
            ax.set_title(comp_ylabel, fontsize=LABEL_FS - 4, fontweight="bold")
            ax.set_xlabel(x_label, fontsize=LABEL_FS - 4)
            ax.set_ylabel("Capacity", fontsize=LABEL_FS - 4)
            _ax_style(ax)

fig.tight_layout()
_save_fig(fig, "sensitivity_components_bundled.png")
plt.show()

plt.close("all")
print("\nSensitivity analysis complete.")
