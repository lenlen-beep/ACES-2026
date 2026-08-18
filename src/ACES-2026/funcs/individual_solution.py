"""
Benchmark: District Heating (DH) vs. Decentralized Heating  --  Report Section 5.5.

Compares the optimized district-heating system (central heat pump + storages + gas
backup + PV + pipe network, produced by main.py / funcs.LCOH) against a per-household
decentralized reference system, on a fair Levelized-Cost-of-Heat (LCOH) basis:

    D1  individual air-to-water heat pump  (+ electric backup heater, bivalent)

Design principle -- SAME useful heat, SAME boundary conditions:
    * reference quantity  Q_use  = sum of the (network-connected) building profiles,
      NOT the generated heat, so that DH network losses stay in the DH numerator;
    * same reference year (2019 load + weather), same annuity factor a (r, n from
      parameters.yaml), same electricity/gas price basis, same CO2 factors,
      same building set (the households actually connected to the modelled trasse).

NON-INVASIVE BY DESIGN
    This module changes no existing file. In particular the grid capacity charge
    (Leistungspreis, decision O-2) is added *symmetrically inside this module* to both
    systems -- funcs/LCOH.py (the DH side) is left untouched. Shared parameters are read
    from parameters.yaml when PyYAML is available and otherwise fall back to the values
    documented in SHARED_DEFAULTS below (identical to parameters.yaml at time of writing).

MODELING DECISIONS (see Report/Section_5-5_Benchmark_Plan.md, "Open Points")
    O-1 Tariff perspective : PRIMARY  = uniform operator-grade procurement price for both
                             systems (isolates technical & scale effects);
                             SENSITIVITY = realistic end-customer tariffs (household retail
                             for decentral vs. RLM wholesale + Leistungspreis for DH).
    O-2 Grid capacity charge: applied symmetrically, each system on its own peak grid
                             withdrawal, bracket chosen by its full-load hours (VBH).
    O-3 HP sizing          : bivalent -- HP to f_biv * design peak, electric heater for the
                             rest; monovalent available as sensitivity (f_biv = 1.0).

Run standalone (uses persisted data + weather cache; ERA5 COP if the project env is present):

    python -m funcs.individual_solution

or call run_benchmark(...) / compare_systems(...) from a session that has just run main.py.

Cost data for the decentral units follow the Danish Energy Agency "Technology Data for
Individual Heating Plants" and the German "Technikkatalog Wärmeplanung" -- the same
catalogue family Section 4.4 already cites for the central components.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless: always save, never require a display
import matplotlib.pyplot as plt

# --- repo paths (this file lives in <repo>/src/ACES-2026/funcs/) -------------------------
_FUNCS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_FUNCS_DIR)                 # .../src/ACES-2026
_REPO_ROOT = os.path.dirname(os.path.dirname(_SRC_DIR))  # repo root
DATA_DIR   = os.path.join(_SRC_DIR, "Data")
PLOTS_DIR  = os.path.join(_SRC_DIR, "plots")


# =========================================================================================
# 1. Shared parameters (identical to parameters.yaml; overridden from it when PyYAML is present)
# =========================================================================================

SHARED_DEFAULTS = {
    "interest_rate": 0.05,          # invest_parameters
    "lifetime_years": 20,
    "supply_temperature": 80.0,     # net_parameters, DH design supply temp [°C]
    "eta_carnot": 0.45,             # HP.eta_carnot (Carnot quality grade)
    "cop_min": 1.5, "cop_max": 7.0,
    "eta_gas_boiler": 0.98,
    # electricity all-in energy price (ct/kWh -> €/MWh); "usual_mid" tariff
    "elec_price_eur_mwh": 167.7,    # = 16.77 ct/kWh
    # grid capacity charge (Leistungspreis) and per-kWh commodity, two VBH brackets
    "cap_charge_high": 115.61, "commodity_high": 13.8,   # €/kW·a ; €/MWh  (>=2500 VBH)
    "cap_charge_low":   25.00, "commodity_low":  49.7,   # €/kW·a ; €/MWh  (<2500 VBH)
    "gas_price_eur_mwh": 72.0,      # 7.2 ct/kWh all-in incl. CO2 (as in Report §3.4.2)
    "pipe_specific_cost": 1000.0,   # €/m (DH network CAPEX)
    # emission factors
    "co2_gas_t_per_mwh": 0.1814,    # Report §3.4.2 (natural gas, gross calorific)
    "co2_elec_t_per_mwh": 0.363,    # German grid mix ~2024 (UBA); assumption, see text
}


def _load_shared():
    """parameters.yaml overrides the defaults where possible; pure-stdlib fallback otherwise."""
    p = dict(SHARED_DEFAULTS)
    try:
        import yaml  # PyYAML only available in the project env
        with open(os.path.join(_SRC_DIR, "parameters.yaml")) as f:
            y = yaml.safe_load(f) or {}
        p["interest_rate"]     = y["invest_parameters"]["interest_rate"]
        p["lifetime_years"]    = y["invest_parameters"]["lifetime_years"]
        p["supply_temperature"] = y["net_parameters"]["supply_temperature"]
        p["eta_carnot"]        = y["system_parameters"]["HP"]["eta_carnot"]
        p["eta_gas_boiler"]    = y["system_parameters"]["gas_boiler"]["eta_gas_boiler"]
        p["elec_price_eur_mwh"] = y["price_parameters"]["electricity"]["tarif"]["usual_mid"] * 10
        p["gas_price_eur_mwh"]  = y["price_parameters"]["gas"]["tarif"]["usual_mid"] * 10
        nc = y["price_parameters"]["electricity"]["network_charge"]
        p["cap_charge_high"] = nc["higher_2500VBH"]["capacity_charge"]
        p["commodity_high"]  = nc["higher_2500VBH"]["commodity_charge"] * 10
        p["cap_charge_low"]  = nc["lower_2500VBH"]["capacity_charge"]
        p["commodity_low"]   = nc["lower_2500VBH"]["commodity_charge"] * 10
        p["pipe_specific_cost"] = y["pipe_parameters"]["specific_invest_pipe"]
    except Exception as e:  # noqa: BLE001  -- yaml missing or key layout changed
        warnings.warn(f"parameters.yaml not read ({e}); using SHARED_DEFAULTS.")
    return p


SHARED = _load_shared()


def annuity_factor(r: float | None = None, n: int | None = None) -> float:
    """Capital recovery factor a = r(1+r)^n / ((1+r)^n - 1). Same definition as the model."""
    r = SHARED["interest_rate"] if r is None else r
    n = SHARED["lifetime_years"] if n is None else n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


# =========================================================================================
# 2. Decentral technology cost & sizing assumptions (DEA / German Technikkatalog Wärmeplanung)
# =========================================================================================

@dataclass
class DecentralParams:
    # --- individual air-to-water heat pump (D1) ---
    hp_offset_eur: float = 3000.0          # base cost per installation (planning, hydraulics)
    hp_specific_eur_kw: float = 1700.0     # €/kW_th installed (small units; > central 1200 €/kW)
    heater_specific_eur_kw: float = 150.0  # €/kW_th electric resistance backup
    buffer_eur: float = 1500.0             # small DHW/decoupling buffer per house
    f_biv: float = 0.60                    # bivalence fraction of the design peak
    design_peak_quantile: float = 0.99     # robust design peak (ignore rare DHW spikes)
    # --- shared ---
    om_rate: float = 0.015                  # O&M as fraction of CAPEX per year (as in DH LCOH)
    # --- retail sensitivity (O-1 variant) ---
    elec_retail_eur_mwh: float = 250.0      # household/HP retail all-in (no separate Leistungspreis)

# design supply temperature by construction year (radiator design temps by era) -- documented
def supply_temperature_for_year(year: float) -> float:
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return 65.0
    if year < 1949:   return 70.0
    if year < 1979:   return 65.0
    if year < 1995:   return 55.0
    if year < 2010:   return 50.0
    return 45.0


# =========================================================================================
# 3. COP and ambient temperature
# =========================================================================================

def carnot_cop(T_amb_C, T_supply_C, eta_carnot=None, cop_min=None, cop_max=None):
    """Temperature-dependent COP (Carnot approach), identical formula to era5_weather.compute_cop.
    T_amb_C: array [°C]; T_supply_C: scalar or array [°C] (broadcast)."""
    eta = SHARED["eta_carnot"] if eta_carnot is None else eta_carnot
    lo  = SHARED["cop_min"] if cop_min is None else cop_min
    hi  = SHARED["cop_max"] if cop_max is None else cop_max
    T_vl = np.asarray(T_supply_C, float) + 273.15
    T_a  = np.asarray(T_amb_C, float) + 273.15
    cop = eta * T_vl / np.maximum(T_vl - T_a, 1e-6)
    return np.clip(cop, lo, hi)


def load_ambient_temperature(year=2019, n_expected=8760):
    """Hourly ambient temperature [°C]. Prefers ERA5 (project env, same source as the model);
    falls back to the meteostat weather cache CSV so the module also runs in a minimal env.
    Returns (T_amb[np.ndarray, len n_expected], source_str)."""
    # (a) ERA5 -- identical source to the optimization/COP in main.py
    try:
        from funcs.era5_weather import load_era5_weather, LAT, LON
        w = load_era5_weather(year, lat=LAT, lon=LON)
        T = w["T_amb_C"].to_numpy()
        return _fit_length(T, n_expected), "ERA5 (funcs.era5_weather)"
    except Exception:
        pass
    # (b) meteostat weather cache CSV (temp column, °C)
    import glob
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "weather_cache", f"weather_*_{year}.csv"))):
        try:
            w = pd.read_csv(path)
            T = pd.to_numeric(w["temp"], errors="coerce").interpolate().bfill().ffill().to_numpy()
            return _fit_length(T, n_expected), f"weather_cache ({os.path.basename(path)})"
        except Exception:
            continue
    raise FileNotFoundError("No ambient temperature source found (ERA5 nor weather_cache).")


def _fit_length(a, n):
    a = np.asarray(a, float)
    if len(a) == n:
        return a
    if len(a) > n:
        return a[:n]
    return np.pad(a, (0, n - len(a)), mode="edge")


# =========================================================================================
# 4. Decentralized systems (vectorized over the building set)
# =========================================================================================

def _capacity_charge(E_el_mwh, peak_el_kw):
    """Grid capacity charge [€/a] on the peak grid withdrawal; bracket chosen by full-load
    hours VBH = annual energy / peak (>=2500 h -> high Leistungspreis / low commodity)."""
    peak_el_kw = np.asarray(peak_el_kw, float)
    E_el_kwh = np.asarray(E_el_mwh, float) * 1000.0
    vbh = np.divide(E_el_kwh, peak_el_kw, out=np.zeros_like(peak_el_kw), where=peak_el_kw > 0)
    high = vbh >= 2500
    return np.where(high, SHARED["cap_charge_high"], SHARED["cap_charge_low"]) * peak_el_kw


def individual_heat_pumps(load_kw, T_amb, T_supply, dp: DecentralParams,
                          tariff_mode="uniform", monovalent=False):
    """D1 -- one air-to-water HP (+ electric heater) per building. load_kw: (T, B) matrix [kW]."""
    a = annuity_factor()
    T_supply = np.asarray(T_supply, float)
    cop = carnot_cop(T_amb[:, None], T_supply[None, :])          # (T, B)

    design_peak = np.quantile(load_kw, dp.design_peak_quantile, axis=0)   # (B,)
    true_peak   = load_kw.max(axis=0)
    f_biv = 1.0 if monovalent else dp.f_biv
    hp_cap   = f_biv * design_peak
    heater_cap = np.maximum(true_peak - hp_cap, 0.0)

    q_hp     = np.minimum(load_kw, hp_cap[None, :])
    q_heater = load_kw - q_hp
    p_el     = q_hp / cop + q_heater                             # electric power [kW] (heater COP=1)

    Q      = load_kw.sum(axis=0) / 1000.0                        # useful heat per building [MWh]
    E_el   = p_el.sum(axis=0) / 1000.0                           # electricity per building [MWh]
    jaz    = np.divide(Q, E_el, out=np.zeros_like(Q), where=E_el > 0)
    peak_el = p_el.max(axis=0)                                   # [kW] for the capacity charge

    capex = (dp.hp_offset_eur + dp.hp_specific_eur_kw * hp_cap
             + dp.heater_specific_eur_kw * heater_cap + dp.buffer_eur)
    price = SHARED["elec_price_eur_mwh"] if tariff_mode == "uniform" else dp.elec_retail_eur_mwh
    opex_energy = price * E_el
    cap_charge = _capacity_charge(E_el, peak_el) if tariff_mode == "uniform" else 0.0
    opex_om = dp.om_rate * capex
    cost_a = capex * a + opex_energy + cap_charge + opex_om

    df = pd.DataFrame({
        "Q_use_MWh": Q, "hp_cap_kW": hp_cap, "heater_cap_kW": heater_cap,
        "T_supply_C": T_supply, "JAZ": jaz, "E_el_MWh": E_el, "peak_el_kW": peak_el,
        "capex_eur": capex, "opex_energy_eur_a": opex_energy,
        "cap_charge_eur_a": cap_charge, "opex_om_eur_a": opex_om, "cost_eur_a": cost_a,
    })
    agg = _aggregate(df, sum_peaks=float(true_peak.sum()),
                     co2=float(E_el.sum() * SHARED["co2_elec_t_per_mwh"]))
    return df, agg


def _aggregate(df, sum_peaks, co2):
    Q = df["Q_use_MWh"].sum()
    capex = df["capex_eur"].sum()
    a = annuity_factor()
    return {
        "Q_use_MWh": Q,
        "capex_total_eur": capex,
        "capex_annualized_eur_a": capex * a,
        "opex_energy_eur_a": df["opex_energy_eur_a"].sum(),
        "cap_charge_eur_a": df["cap_charge_eur_a"].sum(),
        "opex_om_eur_a": df["opex_om_eur_a"].sum(),
        "cost_eur_a": df["cost_eur_a"].sum(),
        "lcoh_eur_mwh": df["cost_eur_a"].sum() / Q,
        "sum_individual_peaks_kW": sum_peaks,
        "installed_th_kW": float(df["hp_cap_kW"].sum()) if "hp_cap_kW" in df.columns else None,
        "mean_JAZ": float(np.average(df["JAZ"], weights=df["Q_use_MWh"]))
                    if "JAZ" in df else None,
        "co2_t_a": co2,
    }


# =========================================================================================
# 5. District-heating reference (from the group's optimization / persisted network result)
# =========================================================================================

@dataclass
class DHReference:
    """DH side of the benchmark. lcoh_total_eur_mwh is the group's optimization result
    (Report §3.4.1 / §5.2). Network CAPEX is deterministic from the trasse length and pipe
    cost, so it is split out explicitly to expose the heat-density effect."""
    lcoh_total_eur_mwh: float = 319.83      # reported model LCOH, §5.3 reference case (incl. grid capacity charge)
    Q_use_mwh: float = 2099.3               # useful heat delivered to connected buildings
    Q_gen_mwh: float = 2099.3               # generated heat at the plant (>= Q_use if losses)
    network_length_m: float = 6406.0
    network_thermal_peak_kw: float = 615.3  # diversified network peak (coincident)
    peak_cop: float = 2.5                   # COP at the design peak (cold hour) for el. peak est.

    def network_capex_eur_mwh(self):
        cap = SHARED["pipe_specific_cost"] * self.network_length_m * annuity_factor()
        return cap / self.Q_use_mwh

    def capacity_charge_eur_mwh(self):
        peak_el = self.network_thermal_peak_kw / self.peak_cop
        # DH runs many hours -> high-VBH bracket
        return (SHARED["cap_charge_high"] * peak_el) / self.Q_use_mwh

    def breakdown_eur_mwh(self):
        """LCOH split (per useful MWh): network CAPEX, capacity charge, and the residual
        generation+operation term implied by the reported total."""
        # rescale the reported total (per generated MWh) to per useful MWh
        total_use = self.lcoh_total_eur_mwh * self.Q_gen_mwh / self.Q_use_mwh
        net = self.network_capex_eur_mwh()
        cap = self.capacity_charge_eur_mwh()
        residual = total_use - net            # generation+storage+opex+O&M
        return {                              # (capacity charge is already inside the reported LCOH, §5.3)
            "network_capex": net,
            "generation_operation": residual,
            "capacity_charge_added": cap,
            "total_incl_cap_charge": total_use,   # capacity charge already in reported LCOH (§5.3); not added on top
            "total_reported_use_basis": total_use,
        }


# =========================================================================================
# 6. Comparison, plots, break-even
# =========================================================================================

def compare_systems(dh: DHReference, d1_agg):
    rows = []
    dhb = dh.breakdown_eur_mwh()
    rows.append({"system": "DH (central)", "lcoh_eur_mwh": dhb["total_incl_cap_charge"],
                 "capex_eur_mwh": dhb["network_capex"] + max(dhb["generation_operation"], 0) * 0.5,
                 "network_capex_eur_mwh": dhb["network_capex"],
                 "energy_opex_eur_mwh": None,
                 "cap_charge_eur_mwh": dhb["capacity_charge_added"],
                 "co2_t_a": None,
                 "mean_JAZ": None, "installed_th_kW": None})
    for name, ag in [("D1 individual HP", d1_agg)]:
        Q = ag["Q_use_MWh"]
        rows.append({
            "system": name, "lcoh_eur_mwh": ag["lcoh_eur_mwh"],
            "capex_eur_mwh": ag["capex_annualized_eur_a"] / Q,
            "network_capex_eur_mwh": 0.0,
            "energy_opex_eur_mwh": ag["opex_energy_eur_a"] / Q,
            "cap_charge_eur_mwh": ag["cap_charge_eur_a"] / Q,
            "co2_t_a": ag["co2_t_a"], "mean_JAZ": ag["mean_JAZ"],
            "installed_th_kW": ag["installed_th_kW"],
        })
    return pd.DataFrame(rows)


def plot_lcoh_comparison(cmp_df, dh: DHReference, n_buildings=None, out=None, show=False):
    """Stacked LCOH bars: DH vs D1 vs D2 (network CAPEX, generation/energy, capacity charge)."""
    systems = cmp_df["system"].tolist()
    x = np.arange(len(systems))
    net = cmp_df["network_capex_eur_mwh"].fillna(0).to_numpy()
    cap = cmp_df["cap_charge_eur_mwh"].fillna(0).to_numpy()
    total = cmp_df["lcoh_eur_mwh"].to_numpy()
    rest = np.maximum(total - net - cap, 0)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.bar(x, rest, 0.55, label="Generation / energy / O&M", color="#00395B")
    ax.bar(x, cap, 0.55, bottom=rest, label="Grid capacity charge", color="#888888")
    ax.bar(x, net, 0.55, bottom=rest + cap, label="Network CAPEX (pipes)", color="#A0463A")
    for xi, t in zip(x, total):
        ax.text(xi, t + 4, f"{t:.0f}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(systems)
    ax.set_ylabel("LCOH  [€/MWh useful heat]")
    nb = f", {n_buildings} buildings" if n_buildings else ""
    ax.set_title(f"District heating vs. decentralized heating  "
                 f"(q_L = {dh.Q_use_mwh/dh.network_length_m:.2f} MWh/(m·a){nb})")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out or os.path.join(PLOTS_DIR, "benchmark_lcoh_comparison.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot saved: {out}")
    if show: plt.show()
    plt.close(fig)
    return out


def breakeven_heat_density(dh: DHReference, competitor_lcoh_eur_mwh, densities=None):
    """LCOH_DH as a function of linear heat density q_L (only the network term scales with it);
    returns the DataFrame and the break-even density vs. the given competitor."""
    if densities is None:
        densities = np.linspace(0.15, 4.0, 60)
    b = dh.breakdown_eur_mwh()
    fixed = b["total_incl_cap_charge"] - b["network_capex"]     # non-network €/MWh
    pipe_per_m = SHARED["pipe_specific_cost"] * annuity_factor()  # €/(m·a)
    # network €/MWh = pipe_per_m / q_L      (since q_L = Q_use / L)
    lcoh_dh = fixed + pipe_per_m / densities
    df = pd.DataFrame({"q_L_MWh_per_m_a": densities, "lcoh_DH_eur_mwh": lcoh_dh})
    be = pipe_per_m / max(competitor_lcoh_eur_mwh - fixed, 1e-9) if competitor_lcoh_eur_mwh > fixed else np.nan
    return df, be


def plot_breakeven(dh: DHReference, competitors: dict, out=None, show=False):
    df, _ = breakeven_heat_density(dh, min(competitors.values()))
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.plot(df["q_L_MWh_per_m_a"], df["lcoh_DH_eur_mwh"], color="#A0463A", lw=2.4,
            label="DH LCOH (network scales with density)")
    q_now = dh.Q_use_mwh / dh.network_length_m
    ax.axvline(q_now, color="#00395B", ls="--", lw=1.5, label=f"Jerrishoe q_L = {q_now:.2f}")
    colors = {"D1 individual HP": "#2F6B4F", "D2 individual gas": "#C17A2F"}
    for name, lcoh in competitors.items():
        ax.axhline(lcoh, color=colors.get(name, "#666"), ls=":", lw=1.8, label=f"{name}: {lcoh:.0f}")
        _, be = breakeven_heat_density(dh, lcoh)
        if be == be and 0 < be < 4.5:
            ax.plot([be], [lcoh], "o", color=colors.get(name, "#666"))
    ax.set_xlabel("Linear heat density  q_L  [MWh/(m·a)]")
    ax.set_ylabel("LCOH  [€/MWh]"); ax.set_ylim(0, min(600, df["lcoh_DH_eur_mwh"].max()))
    ax.set_title("Break-even: at what heat density does district heating compete?")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out or os.path.join(PLOTS_DIR, "benchmark_breakeven_heat_density.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot saved: {out}")
    if show: plt.show()
    plt.close(fig)
    return out


# =========================================================================================
# 7. Data loading (connected building set) and orchestration
# =========================================================================================

def load_connected_buildings(year=2019):
    """The households connected to the modelled trasse (ID>0 in the GeoPackage, present in the
    wide profile CSV) -- the SAME set the DH network serves. Returns (load_kw (T,B), meta_df)."""
    import sqlite3
    wide = pd.read_csv(os.path.join(DATA_DIR, "selected_267_profiles_2019_wide.csv"))
    cols_all = [c for c in wide.columns if c != "Datum"]
    con = sqlite3.connect(os.path.join(DATA_DIR, "Trassierung_Jerrishoe.gpkg"))
    ids = {str(r[0]) for r in con.execute(
        'SELECT "ID" FROM "Trassierung_Jerrishoe" WHERE "ID">0').fetchall()}
    con.close()
    conn = sorted(ids & set(cols_all), key=lambda x: int(x))
    load_kw = wide[conn].astype(float).to_numpy()
    meta = pd.read_csv(os.path.join(DATA_DIR, "selected_267_profiles_meta.csv"))
    # wide column "k" (1..267) maps to meta row k-1 (same ordering as in the prototype CSV)
    rows = [int(c) - 1 for c in conn]
    meta_conn = meta.iloc[rows].reset_index(drop=True)
    meta_conn["wide_col"] = conn
    return load_kw, meta_conn


def run_benchmark(dp: DecentralParams | None = None, tariff_mode="uniform",
                  monovalent=False, dh: DHReference | None = None, save=True, show=False):
    dp = dp or DecentralParams()
    load_kw, meta = load_connected_buildings()
    n_h, n_b = load_kw.shape
    T_amb, tsrc = load_ambient_temperature(n_expected=n_h)
    T_supply = np.array([supply_temperature_for_year(y) for y in meta["construction_year"]])

    d1_df, d1 = individual_heat_pumps(load_kw, T_amb, T_supply, dp, tariff_mode, monovalent)

    if dh is None:
        Q_use = load_kw.sum() / 1000.0
        dh = DHReference(Q_use_mwh=Q_use, Q_gen_mwh=Q_use,
                         network_thermal_peak_kw=float(load_kw.sum(axis=1).max()))
    cmp_df = compare_systems(dh, d1)

    # diversity factor on the connected set
    g = dh.network_thermal_peak_kw / d1["sum_individual_peaks_kW"]

    print("\n" + "=" * 74)
    print(f"BENCHMARK  DH vs. decentralized   |  {n_b} connected buildings, {n_h} h")
    print(f"ambient temperature source: {tsrc}")
    print(f"useful heat Q_use = {dh.Q_use_mwh:,.1f} MWh/a | network {dh.network_length_m:,.0f} m "
          f"| q_L = {dh.Q_use_mwh/dh.network_length_m:.3f} MWh/(m·a)")
    print(f"coincidence factor g = P_net/ΣP̂ = {dh.network_thermal_peak_kw:.0f}/"
          f"{d1['sum_individual_peaks_kW']:.0f} = {g:.3f}  (1/g = {1/g:.2f})")
    print(f"tariff perspective: {tariff_mode}   HP sizing: {'monovalent' if monovalent else 'bivalent'}")
    print("-" * 74)
    print(f"{'system':<20}{'LCOH €/MWh':>12}{'CAPEX €/MWh':>13}{'JAZ':>7}{'CO2 t/a':>10}")
    for _, r in cmp_df.iterrows():
        jaz = f"{r['mean_JAZ']:.2f}" if r["mean_JAZ"] == r["mean_JAZ"] and r["mean_JAZ"] else "  -"
        co2 = f"{r['co2_t_a']:,.0f}" if r["co2_t_a"] == r["co2_t_a"] and r["co2_t_a"] is not None else "  -"
        print(f"{r['system']:<20}{r['lcoh_eur_mwh']:>12.1f}{r['capex_eur_mwh']:>13.1f}{jaz:>7}{co2:>10}")
    print("=" * 74)
    dhb = dh.breakdown_eur_mwh()
    print(f"DH breakdown [€/MWh useful]: network CAPEX {dhb['network_capex']:.1f} | "
          f"generation/operation {dhb['generation_operation']:.1f} | "
          f"capacity charge {dhb['capacity_charge_added']:.1f}")

    competitors = {"D1 individual HP": d1["lcoh_eur_mwh"]}
    for name, lcoh in competitors.items():
        _, be = breakeven_heat_density(dh, lcoh)
        print(f"break-even heat density vs. {name}: q_L = {be:.2f} MWh/(m·a)"
              f"   (current {dh.Q_use_mwh/dh.network_length_m:.2f})")

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_lcoh_comparison(cmp_df, dh, n_buildings=n_b, show=show)
        plot_breakeven(dh, competitors, show=show)
        d1_df.assign(system="D1_HP").to_csv(
            os.path.join(DATA_DIR, "individual_solution_per_building_hp.csv"), index=False)
        cmp_df.to_csv(os.path.join(DATA_DIR, "benchmark_summary.csv"), index=False)
        print(f"CSVs saved to {DATA_DIR}")

    return {"comparison": cmp_df, "d1": d1, "dh": dh, "g": g,
            "d1_df": d1_df, "temp_source": tsrc}


if __name__ == "__main__":
    run_benchmark(tariff_mode="uniform", monovalent=False, save=True, show=False)
