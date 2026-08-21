from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from funcs.plots import (COLOR_WP, COLOR_GAS, COLOR_SPEICHER, COLOR_SAISONAL,
                             COLOR_LAST, COLOR_VERLUST, COLOR_PV,
                             LABEL_FONTSIZE, TICK_FONTSIZE, LEGEND_FONTSIZE, TITLE_FONTSIZE,
                             _ppt_style)
except Exception:
    COLOR_WP, COLOR_GAS, COLOR_SPEICHER = "#00395B", "#C17A2F", "#769D7B"
    COLOR_SAISONAL, COLOR_LAST, COLOR_VERLUST, COLOR_PV = "#2F6B4F", "#1A1A1A", "#A0463A", "#C8A84B"
    LABEL_FONTSIZE = TICK_FONTSIZE = LEGEND_FONTSIZE = TITLE_FONTSIZE = 20

    def _ppt_style(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.2, color="#CCCCCC")
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.margins(x=0)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

COLOR_NEUTRAL = "#888888"
FIGSIZE = (16, 9)

_FUNCS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.dirname(_FUNCS_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_SRC_DIR))
DATA_DIR   = os.path.join(_SRC_DIR, "Data")
PLOTS_DIR  = os.path.join(_SRC_DIR, "plots")

TRASSE_FILES = ("Trassierung_Jerrishoe_50pAQ.gpkg", "Trassierung_Jerrishoe.gpkg")
DH_CACHE_FILE = os.path.join(DATA_DIR, "dh_reference.json")


# --------------------------------------------------
# Gemeinsame Parameter (aus parameters.yaml, sonst Fallback)
# --------------------------------------------------

SHARED_DEFAULTS = {
    "interest_rate": 0.05,
    "lifetime_years": 20,
    "supply_temperature": 80.0,     # °C, Netz-Auslegungsvorlauf
    "eta_carnot": 0.45,
    "cop_min": 1.5, "cop_max": 7.0,
    "eta_gas_boiler": 0.98,
    "elec_price_eur_mwh": 167.7,    # = 16.77 ct/kWh all-in
    "cap_charge_high": 200.65, "commodity_high": 23.9,   # Euro/kW*a ; Euro/MWh (>= 2500 Bh)
    "cap_charge_low":   44.70, "commodity_low":  86.3,   # Euro/kW*a ; Euro/MWh (<  2500 Bh)
    "gas_price_eur_mwh": 72.0,
    "pipe_specific_cost": 400.0,    # Euro/m
    "co2_gas_t_per_mwh": 0.1814,
    "co2_elec_t_per_mwh": 0.363,
}


def _load_shared():
    p = dict(SHARED_DEFAULTS)
    try:
        import yaml
        with open(os.path.join(_SRC_DIR, "parameters.yaml")) as f:
            y = yaml.safe_load(f) or {}
        p["interest_rate"]      = y["invest_parameters"]["interest_rate"]
        p["lifetime_years"]     = y["invest_parameters"]["lifetime_years"]
        p["supply_temperature"] = y["net_parameters"]["supply_temperature"]
        p["eta_carnot"]         = y["system_parameters"]["HP"]["eta_carnot"]
        p["eta_gas_boiler"]     = y["system_parameters"]["gas_boiler"]["eta_gas_boiler"]
        p["elec_price_eur_mwh"] = y["price_parameters"]["electricity"]["tarif"]["usual_mid"] * 10
        p["gas_price_eur_mwh"]  = y["price_parameters"]["gas"]["tarif"]["usual_mid"] * 10
        nc = y["price_parameters"]["electricity"]["network_charge"]
        p["cap_charge_high"] = nc["higher_2500VBH"]["capacity_charge"]
        p["commodity_high"]  = nc["higher_2500VBH"]["commodity_charge"] * 10
        p["cap_charge_low"]  = nc["lower_2500VBH"]["capacity_charge"]
        p["commodity_low"]   = nc["lower_2500VBH"]["commodity_charge"] * 10
        p["pipe_specific_cost"] = y["pipe_parameters"]["specific_invest_pipe"]
    except Exception as e:
        warnings.warn(f"parameters.yaml nicht gelesen ({e}); nutze SHARED_DEFAULTS.")
    return p


SHARED = _load_shared()


def annuity_factor(r: float | None = None, n: int | None = None) -> float:
    r = SHARED["interest_rate"] if r is None else r
    n = SHARED["lifetime_years"] if n is None else n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


# --------------------------------------------------
# Dezentrale Technik: Kosten- und Auslegungsannahmen
# --------------------------------------------------

@dataclass
class DecentralParams:
    hp_offset_eur: float = 3000.0           # Euro je Anlage (Planung, Hydraulik)
    hp_specific_eur_kw: float = 1700.0      # Euro/kW_th (Kleinanlagen)
    heater_specific_eur_kw: float = 150.0   # Euro/kW_th Heizstab
    buffer_eur: float = 1500.0              # Euro je Haus, Pufferspeicher
    f_biv: float = 0.60                     # Bivalenzanteil der Auslegungsleistung
    design_peak_quantile: float = 0.99
    buffer_smoothing_h: int = 6             # Puffer als Gleitmittel ueber h Stunden, 1 = aus
    om_rate: float = 0.015
    apply_capacity_charge: bool = True
    elec_retail_eur_mwh: float = 250.0      # Endkundentarif, Sensitivitaet


def supply_temperature_for_year(year: float) -> float:
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return 65.0
    if year < 1949:   return 70.0
    if year < 1979:   return 65.0
    if year < 1995:   return 55.0
    if year < 2010:   return 50.0
    return 45.0


# --------------------------------------------------
# COP, Umgebungstemperatur und Strompreis
# --------------------------------------------------

def carnot_cop(T_amb_C, T_supply_C, eta_carnot=None, cop_min=None, cop_max=None):
    eta = SHARED["eta_carnot"] if eta_carnot is None else eta_carnot
    lo  = SHARED["cop_min"] if cop_min is None else cop_min
    hi  = SHARED["cop_max"] if cop_max is None else cop_max
    T_vl = np.asarray(T_supply_C, float) + 273.15
    T_a  = np.asarray(T_amb_C, float) + 273.15
    cop = eta * T_vl / np.maximum(T_vl - T_a, 1e-6)
    return np.clip(cop, lo, hi)


def _fit_length(a, n):
    a = np.asarray(a, float)
    if len(a) == n:
        return a
    if len(a) > n:
        return a[:n]
    return np.pad(a, (0, n - len(a)), mode="edge")


def load_ambient_temperature(year=2019, n_expected=8760):
    try:
        from funcs.era5_weather import load_era5_weather, LAT, LON
        w = load_era5_weather(year, lat=LAT, lon=LON)
        return _fit_length(w["T_amb_C"].to_numpy(), n_expected), "ERA5 (funcs.era5_weather)"
    except Exception:
        pass

    import glob
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "weather_cache", f"weather_*_{year}.csv"))):
        try:
            w = pd.read_csv(path)
            T = pd.to_numeric(w["temp"], errors="coerce").interpolate().bfill().ffill().to_numpy()
            return _fit_length(T, n_expected), f"weather_cache ({os.path.basename(path)})"
        except Exception:
            continue
    raise FileNotFoundError("Keine Temperaturquelle gefunden (weder ERA5 noch weather_cache).")


def load_electricity_price(n_expected=8760):
    # Gleiche Basis wie die Fernwaermeseite in LCOH.py: Spot 2024 + Arbeitspreisaufschlaege,
    # aufbereitet wie in main.py (Schalttag raus, um 24 h rotiert). Ohne Leistungspreis.
    try:
        from funcs.read_data import read_price_data, read_parameters
        from funcs.energy_system_optimization import elec_volumetric_surcharge
        from funcs.paths import PARAMETERS_FILE
        ref = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=8784, freq="1h"))
        spot = np.asarray(read_price_data(
            path=os.path.join(DATA_DIR, ""),
            filename="Gro_handelspreise_202401010000_202501010000_Stunde.xlsx",
            load_data=ref), dtype=float)
        spot = spot[~((ref.index.month == 2) & (ref.index.day == 29))]
        spot = np.concatenate([spot[24:], spot[:24]])
        price = spot + elec_volumetric_surcharge(read_parameters(PARAMETERS_FILE))
        return _fit_length(price, n_expected), "Spot 2024 + Aufschlaege (wie LCOH.py)"
    except Exception as e:
        warnings.warn(f"Stundenpreise nicht verfuegbar ({type(e).__name__}: {e}); "
                      f"nutze den pauschalen All-in-Tarif.")
        return float(SHARED["elec_price_eur_mwh"]), "All-in-Tarif (Fallback)"


def _resolve_price(tariff_mode, dp: "DecentralParams", n_t):
    if tariff_mode == "retail":
        return float(dp.elec_retail_eur_mwh), "Endkundentarif"
    if tariff_mode == "uniform":
        # All-in-Tarif enthaelt das Netzentgelt bereits -> zusammen mit dem
        # Leistungspreis wird die Netzkomponente doppelt gezaehlt (nur zum Vergleich)
        return float(SHARED["elec_price_eur_mwh"]), "All-in-Tarif"
    return load_electricity_price(n_t)


# --------------------------------------------------
# Dezentrales System (vektorisiert ueber alle Gebaeude)
# --------------------------------------------------

def _capacity_charge(E_el_mwh, peak_el_kw):
    peak_el_kw = np.asarray(peak_el_kw, float)
    E_el_kwh = np.asarray(E_el_mwh, float) * 1000.0
    vbh = np.divide(E_el_kwh, peak_el_kw, out=np.zeros_like(peak_el_kw), where=peak_el_kw > 0)
    return np.where(vbh >= 2500, SHARED["cap_charge_high"], SHARED["cap_charge_low"]) * peak_el_kw


def buffer_smooth(load_kw, hours: int):
    hours = int(hours)
    if hours <= 1:
        return np.asarray(load_kw, float)
    k = np.ones(hours) / hours
    a = np.asarray(load_kw, float)
    pad = hours // 2
    padded = np.pad(a, ((pad, hours - 1 - pad), (0, 0)), mode="wrap")
    return np.vstack([np.convolve(padded[:, j], k, mode="valid") for j in range(a.shape[1])]).T


def individual_heat_pumps(load_kw, T_amb, T_supply, dp: DecentralParams,
                          tariff_mode="spot", monovalent=False):
    a = annuity_factor()
    T_supply = np.asarray(T_supply, float)

    # Gleichzeitigkeitsfaktor bleibt auf der rohen Stundenbasis, damit er mit der
    # Netzspitze aus der pandapipes-Simulation vergleichbar ist
    raw_peak_sum = float(np.asarray(load_kw, float).max(axis=0).sum())

    load_kw = buffer_smooth(load_kw, dp.buffer_smoothing_h)
    cop = carnot_cop(T_amb[:, None], T_supply[None, :])

    design_peak = np.quantile(load_kw, dp.design_peak_quantile, axis=0)
    true_peak   = load_kw.max(axis=0)
    f_biv       = 1.0 if monovalent else dp.f_biv
    hp_cap      = f_biv * design_peak
    heater_cap  = np.maximum(true_peak - hp_cap, 0.0)

    q_hp     = np.minimum(load_kw, hp_cap[None, :])
    q_heater = load_kw - q_hp
    p_el     = q_hp / cop + q_heater        # kW, Heizstab mit COP = 1

    Q       = load_kw.sum(axis=0) / 1000.0  # MWh Nutzwaerme je Gebaeude
    E_el    = p_el.sum(axis=0) / 1000.0     # MWh Strom je Gebaeude
    jaz     = np.divide(Q, E_el, out=np.zeros_like(Q), where=E_el > 0)
    peak_el = p_el.max(axis=0)

    capex = (dp.hp_offset_eur + dp.hp_specific_eur_kw * hp_cap
             + dp.heater_specific_eur_kw * heater_cap + dp.buffer_eur)

    price, _ = _resolve_price(tariff_mode, dp, load_kw.shape[0])
    if np.ndim(price) == 0:
        opex_energy = float(price) * E_el
    else:
        opex_energy = (p_el / 1000.0 * np.asarray(price)[:, None]).sum(axis=0)

    cap_charge = (_capacity_charge(E_el, peak_el)
                  if tariff_mode != "retail" and dp.apply_capacity_charge else 0.0)
    opex_om = dp.om_rate * capex
    cost_a  = capex * a + opex_energy + cap_charge + opex_om

    df = pd.DataFrame({
        "Q_use_MWh": Q, "hp_cap_kW": hp_cap, "heater_cap_kW": heater_cap,
        "T_supply_C": T_supply, "JAZ": jaz, "E_el_MWh": E_el, "peak_el_kW": peak_el,
        "capex_eur": capex, "opex_energy_eur_a": opex_energy,
        "cap_charge_eur_a": cap_charge, "opex_om_eur_a": opex_om, "cost_eur_a": cost_a,
    })
    agg = _aggregate(df, sum_peaks=raw_peak_sum,
                     co2=float(E_el.sum() * SHARED["co2_elec_t_per_mwh"]))
    return df, agg


def _aggregate(df, sum_peaks, co2):
    Q = df["Q_use_MWh"].sum()
    capex = df["capex_eur"].sum()
    return {
        "Q_use_MWh": Q,
        "capex_total_eur": capex,
        "capex_annualized_eur_a": capex * annuity_factor(),
        "opex_energy_eur_a": df["opex_energy_eur_a"].sum(),
        "cap_charge_eur_a": df["cap_charge_eur_a"].sum(),
        "opex_om_eur_a": df["opex_om_eur_a"].sum(),
        "cost_eur_a": df["cost_eur_a"].sum(),
        "lcoh_eur_mwh": df["cost_eur_a"].sum() / Q,
        "sum_individual_peaks_kW": sum_peaks,
        "installed_th_kW": float(df["hp_cap_kW"].sum()),
        "mean_JAZ": float(np.average(df["JAZ"], weights=df["Q_use_MWh"])),
        "co2_t_a": co2,
    }


# --------------------------------------------------
# Fernwaerme-Referenz
# --------------------------------------------------

@dataclass
class DHReference:
    lcoh_total_eur_mwh: float = 185.2       # Euro/MWh eingespeist, Bericht 5.2
    Q_use_mwh: float = 2355.7               # Nutzwaerme in den 134 Gebaeuden
    Q_gen_mwh: float = 3069.4               # Einspeisung am Heizwerk, Bericht 5.1
    network_length_m: float = 6869.0
    network_thermal_peak_kw: float = 805.4  # Spitzenlast am Heizwerk
    peak_cop: float = 2.5
    capacity_charge_reported_eur_mwh_gen: float = 4.58

    def network_capex_eur_mwh(self):
        cap = SHARED["pipe_specific_cost"] * self.network_length_m * annuity_factor()
        return cap / self.Q_use_mwh

    def capacity_charge_eur_mwh(self):
        return self.capacity_charge_reported_eur_mwh_gen * self.Q_gen_mwh / self.Q_use_mwh

    def network_losses_mwh(self):
        return max(self.Q_gen_mwh - self.Q_use_mwh, 0.0)

    def breakdown_eur_mwh(self):
        # gemeldeten LCOH von "je eingespeister MWh" auf "je Nutz-MWh" umrechnen
        total_use = self.lcoh_total_eur_mwh * self.Q_gen_mwh / self.Q_use_mwh
        net = self.network_capex_eur_mwh()
        return {
            "network_capex": net,
            "generation_operation": total_use - net,
            "capacity_charge_added": self.capacity_charge_eur_mwh(),
            "total_incl_cap_charge": total_use,
            "total_reported_use_basis": total_use,
        }


def run_main_pipeline(force: bool = False, cache_file: str = DH_CACHE_FILE) -> dict:
    if not force and os.path.exists(cache_file):
        with open(cache_file) as f:
            d = json.load(f)
        print(f"Fernwaerme-Referenz aus Cache: {os.path.basename(cache_file)}")
        return d

    import runpy
    main_path = os.path.join(_SRC_DIR, "main.py")
    cwd0, show0 = os.getcwd(), plt.show
    try:
        os.chdir(_REPO_ROOT)                    # main.py nutzt Pfade relativ zum Repo-Root
        plt.show = lambda *a, **k: None
        print(f"Starte {main_path} (Netzsimulation + Optimierung, dauert) ...")
        g = runpy.run_path(main_path, run_name="__benchmark__")
    finally:
        plt.show = show0
        os.chdir(cwd0)

    res = g["result_df"]
    d = {
        "lcoh_total_eur_mwh": float(g["lcoh"]),
        "Q_gen_mwh": float(res["load_kW"].sum() / 1000.0),
        "Q_use_mwh": float(res["consumer_load_kW"].sum() / 1000.0),
        "network_length_m": float(g["network_length"]),
        "network_thermal_peak_kw": float(res["load_kW"].max()),
        "capacity_charge_reported_eur_mwh_gen":
            float(g["components"]["Netzentgelt Leistungspreis"]["eur_per_mwh"]),
        "hp_capacity_mw": float(g["result_hp_capacity"]),
        "pv_capacity_mw": float(g["result_pv_capacity"]),
    }
    with open(cache_file, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Fernwaerme-Referenz aus main.py uebernommen, Cache: {cache_file}")
    return d


def dh_from_main(force: bool = False) -> DHReference:
    d = run_main_pipeline(force=force)
    return DHReference(**{k: v for k, v in d.items()
                          if k in DHReference.__dataclass_fields__})


# --------------------------------------------------
# Vergleich, Break-even und Plots
# --------------------------------------------------

def compare_systems(dh: DHReference, d1_agg):
    dhb = dh.breakdown_eur_mwh()
    rows = [{"system": "DH (central)",
             "lcoh_eur_mwh": dhb["total_incl_cap_charge"],
             "capex_eur_mwh": dhb["network_capex"] + max(dhb["generation_operation"], 0) * 0.5,
             "network_capex_eur_mwh": dhb["network_capex"],
             "energy_opex_eur_mwh": None,
             "cap_charge_eur_mwh": dhb["capacity_charge_added"],
             "co2_t_a": None, "mean_JAZ": None, "installed_th_kW": None}]

    Q = d1_agg["Q_use_MWh"]
    rows.append({"system": "D1 individual HP",
                 "lcoh_eur_mwh": d1_agg["lcoh_eur_mwh"],
                 "capex_eur_mwh": d1_agg["capex_annualized_eur_a"] / Q,
                 "network_capex_eur_mwh": 0.0,
                 "energy_opex_eur_mwh": d1_agg["opex_energy_eur_a"] / Q,
                 "cap_charge_eur_mwh": d1_agg["cap_charge_eur_a"] / Q,
                 "co2_t_a": d1_agg["co2_t_a"], "mean_JAZ": d1_agg["mean_JAZ"],
                 "installed_th_kW": d1_agg["installed_th_kW"]})
    return pd.DataFrame(rows)


def breakeven_heat_density(dh: DHReference, competitor_lcoh_eur_mwh, densities=None):
    # Nur der Netzterm skaliert mit der Waermedichte: LCOH_DH(q_L) = fix + Rohrkosten * a / q_L
    if densities is None:
        densities = np.linspace(0.15, 4.0, 60)
    b = dh.breakdown_eur_mwh()
    fixed = b["total_incl_cap_charge"] - b["network_capex"]
    pipe_per_m = SHARED["pipe_specific_cost"] * annuity_factor()
    df = pd.DataFrame({"q_L_MWh_per_m_a": densities,
                       "lcoh_DH_eur_mwh": fixed + pipe_per_m / densities})
    be = (pipe_per_m / max(competitor_lcoh_eur_mwh - fixed, 1e-9)
          if competitor_lcoh_eur_mwh > fixed else np.nan)
    return df, be


def plot_lcoh_comparison(cmp_df, dh: DHReference, n_buildings=None, out=None, show=False):
    systems = cmp_df["system"].tolist()
    x = np.arange(len(systems))
    net = cmp_df["network_capex_eur_mwh"].fillna(0).to_numpy()
    cap = cmp_df["cap_charge_eur_mwh"].fillna(0).to_numpy()
    total = cmp_df["lcoh_eur_mwh"].to_numpy()
    rest = np.maximum(total - net - cap, 0)

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.bar(x, rest, 0.55, label="Generation / energy / O&M", color=COLOR_WP)
    ax.bar(x, cap, 0.55, bottom=rest, label="Grid capacity charge", color=COLOR_NEUTRAL)
    ax.bar(x, net, 0.55, bottom=rest + cap, label="Network CAPEX (pipes)", color=COLOR_VERLUST)
    for xi, t in zip(x, total):
        ax.text(xi, t + 4, f"{t:.0f}", ha="center", va="bottom",
                fontsize=LABEL_FONTSIZE, fontweight="bold", color=COLOR_LAST)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("LCOH in €/MWh useful heat", fontsize=LABEL_FONTSIZE)
    nb = f", {n_buildings} buildings" if n_buildings else ""
    ax.set_title(f"District heating vs. decentralized heating  "
                 f"(q_L = {dh.Q_use_mwh/dh.network_length_m:.2f} MWh/(m·a){nb})",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax)
    ax.margins(x=0.15)
    ax.set_ylim(0, total.max() * 1.18)
    fig.tight_layout()

    out = out or os.path.join(PLOTS_DIR, "benchmark_lcoh_comparison.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_breakeven(dh: DHReference, competitors: dict, out=None, show=False):
    # x-Achse so weit ziehen, dass der Schnittpunkt im Bild liegt
    _, be_min = breakeven_heat_density(dh, min(competitors.values()))
    q_max = 4.0 if be_min != be_min else max(4.0, be_min * 1.25)
    df, _ = breakeven_heat_density(dh, min(competitors.values()),
                                   densities=np.linspace(0.15, q_max, 200))

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.plot(df["q_L_MWh_per_m_a"], df["lcoh_DH_eur_mwh"], color=COLOR_VERLUST, lw=2.4,
            label="DH LCOH (network scales with density)")
    q_now = dh.Q_use_mwh / dh.network_length_m
    ax.axvline(q_now, color=COLOR_LAST, ls="--", lw=1.5, label=f"Jerrishoe q_L = {q_now:.2f}")

    colors = {"D1 individual HP": COLOR_SAISONAL, "D2 individual gas": COLOR_GAS}
    for name, lcoh in competitors.items():
        c = colors.get(name, COLOR_WP)
        ax.axhline(lcoh, color=c, ls=":", lw=1.8, label=f"{name}: {lcoh:.0f} €/MWh")
        _, be = breakeven_heat_density(dh, lcoh)
        if be == be and 0 < be < df["q_L_MWh_per_m_a"].max():
            ax.axvline(be, color=c, ls="--", lw=1.5, label=f"break-even q_L = {be:.2f}")
            ax.plot([be], [lcoh], "o", color=c, markersize=10)

    ax.set_xlabel("Linear heat density q_L in MWh/(m·a)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("LCOH in €/MWh", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0, min(600, df["lcoh_DH_eur_mwh"].max()))
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)

    # Abstand zwischen heutiger Dichte und Break-even als Faktor beschriften
    _, be_ref = breakeven_heat_density(dh, min(competitors.values()))
    if be_ref == be_ref and be_ref > q_now:
        y_ar = ax.get_ylim()[1] * 0.84
        ax.annotate("", xy=(q_now, y_ar), xytext=(be_ref, y_ar),
                    arrowprops=dict(arrowstyle="<->", color=COLOR_LAST, lw=1.5))
        ax.text((q_now + be_ref) / 2, y_ar * 1.03, f"{be_ref/q_now:.1f} ×",
                ha="center", va="bottom", fontsize=LABEL_FONTSIZE,
                fontweight="bold", color=COLOR_LAST)
    _ppt_style(ax)
    fig.tight_layout()

    out = out or os.path.join(PLOTS_DIR, "benchmark_breakeven_heat_density.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


# --------------------------------------------------
# Angeschlossene Gebaeude laden
# --------------------------------------------------

def load_connected_buildings(year=2019, trasse_file=None):
    # Gleiches GeoPackage wie main.py; Layername wird gelesen, weil er sich
    # zwischen den beiden Dateien in der Gross-/Kleinschreibung unterscheidet
    import sqlite3

    wide = pd.read_csv(os.path.join(DATA_DIR, "selected_267_profiles_2019_wide.csv"))
    cols_all = [c for c in wide.columns if c != "Datum"]

    candidates = (trasse_file,) if trasse_file else TRASSE_FILES
    path = next((os.path.join(DATA_DIR, f) for f in candidates
                 if os.path.exists(os.path.join(DATA_DIR, f))), None)
    if path is None:
        raise FileNotFoundError(f"Keine Trassen-Datei in {DATA_DIR} gefunden (geprueft: {candidates}).")

    con = sqlite3.connect(path)
    layer = con.execute("SELECT table_name FROM gpkg_contents").fetchone()[0]
    ids = {str(r[0]) for r in con.execute(f'SELECT "ID" FROM "{layer}" WHERE "ID">0').fetchall()}
    trasse_length_m = float(con.execute(f'SELECT SUM("Length_m") FROM "{layer}"').fetchone()[0])
    con.close()

    conn = sorted(ids & set(cols_all), key=lambda x: int(x))
    load_kw = wide[conn].astype(float).to_numpy()

    meta = pd.read_csv(os.path.join(DATA_DIR, "selected_267_profiles_meta.csv"))
    meta_conn = meta.iloc[[int(c) - 1 for c in conn]].reset_index(drop=True)
    meta_conn["wide_col"] = conn
    meta_conn.attrs["trasse_file"] = os.path.basename(path)
    meta_conn.attrs["trasse_length_m"] = trasse_length_m
    return load_kw, meta_conn


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

def run_benchmark(dp: DecentralParams | None = None, tariff_mode="spot",
                  monovalent=False, dh: DHReference | None = None, save=True, show=False,
                  use_main: bool = False, force_main: bool = False):
    dp = dp or DecentralParams()

    if dh is None and (use_main or force_main):
        try:
            dh = dh_from_main(force=force_main)
        except Exception as e:
            warnings.warn(f"main.py nicht ausfuehrbar ({type(e).__name__}: {e}); "
                          f"nutze die Werte aus dem Bericht.")

    load_kw, meta = load_connected_buildings()
    n_h, n_b = load_kw.shape
    T_amb, tsrc = load_ambient_temperature(n_expected=n_h)
    T_supply = np.array([supply_temperature_for_year(y) for y in meta["construction_year"]])

    d1_df, d1 = individual_heat_pumps(load_kw, T_amb, T_supply, dp, tariff_mode, monovalent)

    if dh is None:
        dh = DHReference()
        L_data = meta.attrs.get("trasse_length_m")
        if L_data:
            dh.network_length_m = float(L_data)
        Q_use_data = float(load_kw.sum() / 1000.0)
        if abs(Q_use_data - dh.Q_use_mwh) / dh.Q_use_mwh > 0.02:
            loss_ratio = dh.Q_gen_mwh / dh.Q_use_mwh    # Verlustanteil des Berichts halten
            warnings.warn(
                f"Gebaeudewaerme ({Q_use_data:,.1f} MWh) weicht vom Berichtsfall "
                f"({dh.Q_use_mwh:,.1f} MWh) ab; Q_use/Q_gen werden skaliert.")
            dh.Q_use_mwh = Q_use_data
            dh.Q_gen_mwh = Q_use_data * loss_ratio

    cmp_df = compare_systems(dh, d1)
    g = dh.network_thermal_peak_kw / d1["sum_individual_peaks_kW"]

    print("\n" + "=" * 74)
    print(f"BENCHMARK  Fernwaerme vs. dezentral   |  {n_b} Gebaeude, {n_h} h")
    print(f"Trasse: {meta.attrs.get('trasse_file', 'n/a')}   Temperatur: {tsrc}")
    print(f"Nutzwaerme Q_use = {dh.Q_use_mwh:,.1f} MWh/a | Netz {dh.network_length_m:,.0f} m "
          f"| q_L = {dh.Q_use_mwh/dh.network_length_m:.3f} MWh/(m*a)")
    print(f"Einspeisung Q_gen = {dh.Q_gen_mwh:,.1f} MWh/a | Netzverluste "
          f"{dh.network_losses_mwh():,.1f} MWh/a "
          f"({dh.network_losses_mwh()/dh.Q_gen_mwh*100:.1f} % der Einspeisung)")
    print(f"Gleichzeitigkeitsfaktor g = {dh.network_thermal_peak_kw:.0f}/"
          f"{d1['sum_individual_peaks_kW']:.0f} = {g:.3f}  (1/g = {1/g:.2f})")
    print(f"Tarifbasis: {tariff_mode}   Auslegung: {'monovalent' if monovalent else 'bivalent'}")
    print("-" * 74)
    print(f"{'System':<20}{'LCOH Eur/MWh':>14}{'CAPEX Eur/MWh':>15}{'JAZ':>7}{'CO2 t/a':>10}")
    for _, r in cmp_df.iterrows():
        jaz = f"{r['mean_JAZ']:.2f}" if r["mean_JAZ"] == r["mean_JAZ"] and r["mean_JAZ"] else "  -"
        co2 = f"{r['co2_t_a']:,.0f}" if r["co2_t_a"] == r["co2_t_a"] and r["co2_t_a"] is not None else "  -"
        print(f"{r['system']:<20}{r['lcoh_eur_mwh']:>14.1f}{r['capex_eur_mwh']:>15.1f}{jaz:>7}{co2:>10}")
    print("=" * 74)

    dhb = dh.breakdown_eur_mwh()
    print(f"Fernwaerme je Nutz-MWh: Netz-CAPEX {dhb['network_capex']:.1f} | "
          f"Erzeugung/Betrieb {dhb['generation_operation']:.1f} | "
          f"Leistungspreis {dhb['capacity_charge_added']:.1f}")

    competitors = {"D1 individual HP": d1["lcoh_eur_mwh"]}
    _, be_ref = breakeven_heat_density(dh, d1["lcoh_eur_mwh"])
    print(f"Break-even Waermedichte: q_L = {be_ref:.2f} MWh/(m*a)   "
          f"(aktuell {dh.Q_use_mwh/dh.network_length_m:.2f})")

    # --------------------------------------------------
    # Varianten der dezentralen Annahmen
    # --------------------------------------------------

    print("-" * 74)
    print(f"{'Variante (Einzel-WP)':<40}{'LCOH':>9}{'JAZ':>7}{'Break-even q_L':>18}")
    for label, kw in [
        ("Referenz (Spot + Aufschlaege + Lp.)", {}),
        ("ohne Pufferglaettung (Stundenpeak)", {"buffer_smoothing_h": 1}),
        ("24 h Puffer", {"buffer_smoothing_h": 24}),
        ("ohne Leistungspreis", {"apply_capacity_charge": False}),
        ("monovalent (ohne Heizstab)", {"_mono": True}),
        ("All-in-Tarif 16.77 ct/kWh", {"_mode": "uniform"}),
        ("Endkundentarif 250 Eur/MWh", {"_mode": "retail"}),
    ]:
        mono = kw.pop("_mono", monovalent)
        mode = kw.pop("_mode", tariff_mode)
        dpv = DecentralParams(**{**asdict(dp), **kw})
        _, agg = individual_heat_pumps(load_kw, T_amb, T_supply, dpv, mode, mono)
        _, be = breakeven_heat_density(dh, agg["lcoh_eur_mwh"])
        # NaN = Konkurrent liegt unter den Fixkosten der Fernwaerme, dann gibt es keinen Schnittpunkt
        be_txt = f"{be:.2f}" if be == be else "unerreichbar"
        print(f"{label:<40}{agg['lcoh_eur_mwh']:>9.1f}{agg['mean_JAZ']:>7.2f}{be_txt:>18}")

    # --------------------------------------------------
    # Sensitivitaet ueber die Waermedichte
    # --------------------------------------------------

    sweep, _ = breakeven_heat_density(dh, d1["lcoh_eur_mwh"],
                                      densities=np.array([0.25, 0.34, 0.5, 0.75, 1.0,
                                                          1.5, 2.0, 3.0, 5.0]))
    print("-" * 74)
    print(f"Waermedichte-Sweep (Einzel-WP = {d1['lcoh_eur_mwh']:.1f} Eur/MWh)")
    print(f"{'q_L [MWh/(m*a)]':<20}{'LCOH_DH':>10}{'Delta':>20}")
    for _, r in sweep.iterrows():
        print(f"{r['q_L_MWh_per_m_a']:<20.2f}{r['lcoh_DH_eur_mwh']:>10.1f}"
              f"{r['lcoh_DH_eur_mwh'] - d1['lcoh_eur_mwh']:>+19.1f}")
    print("=" * 74)

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_lcoh_comparison(cmp_df, dh, n_buildings=n_b, show=show)
        plot_breakeven(dh, competitors, show=show)
        d1_df.assign(system="D1_HP").to_csv(
            os.path.join(DATA_DIR, "individual_solution_per_building_hp.csv"), index=False)
        cmp_df.to_csv(os.path.join(DATA_DIR, "benchmark_summary.csv"), index=False)
        full_sweep, _ = breakeven_heat_density(dh, d1["lcoh_eur_mwh"])
        full_sweep.assign(lcoh_individual_hp_eur_mwh=d1["lcoh_eur_mwh"]).to_csv(
            os.path.join(DATA_DIR, "benchmark_heat_density_sweep.csv"), index=False)
        print(f"CSVs gespeichert in {DATA_DIR}")

    return {"comparison": cmp_df, "d1": d1, "dh": dh, "g": g,
            "d1_df": d1_df, "temp_source": tsrc}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Benchmark Fernwaerme vs. dezentral (Bericht 5.5).")
    ap.add_argument("--use-main", action="store_true")
    ap.add_argument("--force-main", action="store_true")
    ap.add_argument("--monovalent", action="store_true")
    ap.add_argument("--tariff", default="spot", choices=["spot", "uniform", "retail"])
    args = ap.parse_args()

    run_benchmark(tariff_mode=args.tariff, monovalent=args.monovalent, save=True, show=False,
                  use_main=args.use_main, force_main=args.force_main)
