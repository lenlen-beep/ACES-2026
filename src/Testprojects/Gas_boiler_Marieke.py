"""
Dispatch Optimization: Gas Boiler + Heat Pump for a Small Municipality
=======================================================================

This script solves a cost-minimizing dispatch optimization problem for a
small district heating system (~2,000 inhabitants, ~2.6 % of Flensburg's
total heat demand) served by two complementary technologies: Gas Boiler and Heat Pump  

The optimization is formulated as a Linear Program (LP) using the Pyomo
modelling framework and solved with the open-source HiGHS solver. For each
of the 8,759 hours of the year the model determines the cost-optimal split
of heat production between the two plants, subject to:

  1. Heat balance  – combined output must always meet the hourly demand.
  2. Capacity limits – each plant is bounded by its installed capacity.
  3. Boiler lock   – the gas boiler is only permitted to operate when it
                     produces heat more cheaply than the heat pump, OR when
                     heat pump capacity alone is insufficient to cover demand.

Input data
----------
  - Hourly heat demand        : Flensburg 2017, scaled by factor 0.026
                                (source: Flensburg2017_Municipality_scale026.xlsx)
  - Hourly electricity prices : Germany / Luxembourg day-ahead market, 2024
                                (source: Gro_handelspreise_202401010000_202501010000_Stunde.xlsx)
  - Monthly gas prices        : Germany 2017
                                (source: Gas_Prices_Germany_2017.xlsx)

Outputs
-------
  - Console summary  : total annual costs (€) and operating hours per mode
  - dispatch_optimization_result.png  : full-year dispatch, price signals,
                                        and hourly operating-mode chart
  - dispatch_analysis.png             : monthly cost breakdown, monthly
                                        operating hours, and load duration curve

                                        
To Do
-----
CO2 Preise, unterscheiden zwischen Erdgas und Biogas
Sensitivity analysis




"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ─────────────────────────────────────────────
# 1. Load demand (Flensburg 2017, hourly) scaled to 2.6 % (peak ~9 MW) --> 2000 people municipality
# ─────────────────────────────────────────────
dem_df  = pd.read_excel("Flensburg2017_Municipality_scale026.xlsx", sheet_name="Hourly Data")
demand  = dem_df["Q_Municipality (MW)"].values          # MW_th
months  = dem_df["Month"].values                        # 1–12, for gas price lookup
N                  = len(demand)                        # 8 759 hours
T                  = range(N)
peak_demand        = float(np.max(demand))              # MW_th, Spitzenwärmelast aus Daten
annual_demand_mwh  = float(np.sum(demand))              # MWh_th/a, Jahreswärmebedarf

# Legal cap: gas boiler ≤ 10 % of annual heat feed-in (GEG / WPG, Stand Juni 2026)
BOILER_SHARE_MAX   = 0.10
boiler_energy_cap  = BOILER_SHARE_MAX * annual_demand_mwh   # MWh_th/a

# ─────────────────────────────────────────────
# 2. Load electricity prices (DE 2024, hourly, index-aligned)
# ─────────────────────────────────────────────
el_df       = pd.read_excel("Gro_handelspreise_202401010000_202501010000_Stunde.xlsx", skiprows=9)
el_price    = el_df["Deutschland/Luxemburg [€/MWh]"].values[:N]   # €/MWh_el

# ─────────────────────────────────────────────
# 3. Load gas prices (Germany 2017, monthly → hourly) (daily prices not available for free market)
# ─────────────────────────────────────────────
gas_df      = pd.read_excel("Gas_Prices_Germany_2017.xlsx", skiprows=2)
gas_monthly = {i + 1: row["End Price [€/MWh]"] for i, row in gas_df.iloc[:12].iterrows()}
gas_price_fuel = np.array([gas_monthly[m] for m in months])       # €/MWh_fuel

# ─────────────────────────────────────────────
# 4. Plant parameters
# ─────────────────────────────────────────────
ETA_BOILER   = 0.98   # thermal efficiency of gas boiler
COP_HP       = 3.5    # COP of heat pump
P_HP_MAX     = 8.0    # MW_th  (base-load sizing ~55 % of peak)
P_BOILER_MAX = float(np.ceil(peak_demand))             # MW_th, Spitzenlastkessel: Auslegung auf Spitzenwärmelast

# Specific investment costs as function of plant size (economies of scale)
# Power law:  c_spec(P) = a * P_MW^b  [€/MW_th],  b < 0 (larger → cheaper per MW)
# Calibration points (literature, DE Nahwärme):
#   Gas boiler : 150 €/kW at 0.5 MW  |  75 €/kW at 10 MW
#   Heat pump  : 1200 €/kW at 0.5 MW | 500 €/kW at 10 MW

def capex_specific(P_MW, a, b):
    """Specific investment cost [€/MW_th] as a function of installed capacity."""
    return a * P_MW ** b

# Scaling coefficients (a [€/MW_th], b [-]) – derived from calibration points above
_SCALE_BOILER = (127_800, -0.231)
_SCALE_HP     = (980_400, -0.292)

CAPEX_BOILER = capex_specific(P_BOILER_MAX, *_SCALE_BOILER)   # €/MW_th
CAPEX_HP     = capex_specific(P_HP_MAX,     *_SCALE_HP)       # €/MW_th

# Economic parameters for annuity calculation
INTEREST_RATE    = 0.05   # 5 % p.a.
LIFETIME_BOILER  = 20     # Jahre
LIFETIME_HP      = 20     # Jahre

def annuity_factor(i, n):
    """Annuitätenfaktor: i*(1+i)^n / ((1+i)^n - 1)"""
    return i * (1 + i) ** n / ((1 + i) ** n - 1)

annuity_boiler = annuity_factor(INTEREST_RATE, LIFETIME_BOILER)
annuity_hp     = annuity_factor(INTEREST_RATE, LIFETIME_HP)

# Total and annualised investment costs
invest_boiler        = CAPEX_BOILER * P_BOILER_MAX   # € gesamt
invest_hp            = CAPEX_HP     * P_HP_MAX        # € gesamt
total_invest         = invest_boiler + invest_hp       # € gesamt

annualised_boiler    = annuity_boiler * invest_boiler  # €/a
annualised_hp        = annuity_hp     * invest_hp      # €/a
total_annualised     = annualised_boiler + annualised_hp  # €/a

# Effective cost per produced MWh_th
gas_cost_th = gas_price_fuel / ETA_BOILER          # €/MWh_th
hp_cost_th  = el_price / COP_HP                    # €/MWh_th

# Pre-dispatch decision: boiler only allowed when cheaper than heat pump,
# OR when heat pump alone cannot cover demand (capacity fallback).
boiler_allowed = np.where(
    (gas_cost_th <= hp_cost_th) | (demand > P_HP_MAX),
    1, 0
)

# ─────────────────────────────────────────────
# 5. Pyomo model
# ─────────────────────────────────────────────
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)

model.P_boiler = pyo.Var(model.T, bounds=(0, P_BOILER_MAX))
model.P_hp     = pyo.Var(model.T, bounds=(0, P_HP_MAX))

def obj_rule(m):
    cost_boiler = sum(float(gas_cost_th[t]) * m.P_boiler[t] for t in m.T)
    cost_hp     = sum(float(hp_cost_th[t])  * m.P_hp[t]     for t in m.T)
    return cost_boiler + cost_hp

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

def heat_balance(m, t):
    return m.P_boiler[t] + m.P_hp[t] >= float(demand[t])

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

def boiler_lock(m, t):
    return m.P_boiler[t] <= float(boiler_allowed[t]) * P_BOILER_MAX

model.boiler_lock = pyo.Constraint(model.T, rule=boiler_lock)

# Legal cap: annual boiler output ≤ 10 % of total heat supply (GEG / WPG)
def boiler_annual_cap(m):
    return sum(m.P_boiler[t] for t in m.T) <= boiler_energy_cap

model.boiler_annual_cap = pyo.Constraint(rule=boiler_annual_cap)

# ─────────────────────────────────────────────
# 6. Solve
# ─────────────────────────────────────────────
solver  = pyo.SolverFactory("highs")
results = solver.solve(model)

# ─────────────────────────────────────────────
# 7. Extract results
# ─────────────────────────────────────────────
P_boiler_res = np.array([pyo.value(model.P_boiler[t]) for t in T])
P_hp_res     = np.array([pyo.value(model.P_hp[t])     for t in T])

total_cost_boiler = float(np.sum(gas_cost_th * P_boiler_res))
total_cost_hp     = float(np.sum(hp_cost_th  * P_hp_res))
total_cost        = total_cost_boiler + total_cost_hp

boiler_energy_used = float(np.sum(P_boiler_res))        # MWh_th/a tatsächlich
boiler_share_pct   = boiler_energy_used / annual_demand_mwh * 100

print(f"\n── Plant sizing ───────────────────────────")
print(f"  Peak heat demand                        : {peak_demand:>10.2f} MW_th")
print(f"  Annual heat demand                      : {annual_demand_mwh:>10.0f} MWh_th/a")
print(f"  Gas boiler (Spitzenlast, ceil)          : {P_BOILER_MAX:>10.0f} MW_th")
print(f"  Heat pump  (Grundlast, fix)             : {P_HP_MAX:>10.0f} MW_th")

print(f"\n── Legal cap (GEG/WPG): boiler ≤ {BOILER_SHARE_MAX*100:.0f} % ──")
print(f"  Allowed boiler energy                   : {boiler_energy_cap:>10.0f} MWh_th/a")
print(f"  Actual  boiler energy                   : {boiler_energy_used:>10.0f} MWh_th/a")
print(f"  Boiler share                            : {boiler_share_pct:>9.1f} %  {'[BINDING]' if boiler_share_pct >= BOILER_SHARE_MAX*100 - 0.05 else '[slack]'}")

print(f"\n── Investment costs (CAPEX) ───────────────")
print(f"  Gas boiler  ({P_BOILER_MAX:.0f} MW → {CAPEX_BOILER/1e3:.1f} k€/MW_th spez.) : {invest_boiler/1e3:>8.0f} k€")
print(f"  Heat pump   ({P_HP_MAX:.0f} MW → {CAPEX_HP/1e3:.1f} k€/MW_th spez.) : {invest_hp/1e3:>8.0f} k€")
print(f"  Total investment                        : {total_invest/1e3:>10.0f} k€")
print(f"\n── Annualised investment costs (i={INTEREST_RATE*100:.0f}%, {LIFETIME_BOILER}a) ──")
print(f"  Gas boiler  (α={annuity_boiler:.4f})        : {annualised_boiler/1e3:>10.1f} k€/a")
print(f"  Heat pump   (α={annuity_hp:.4f})        : {annualised_hp/1e3:>10.1f} k€/a")
print(f"  Total annualised                        : {total_annualised/1e3:>10.1f} k€/a")

print(f"\n── Operating costs (OPEX) ─────────────────")
print(f"  Gas boiler total : {total_cost_boiler:>12.0f} €")
print(f"  Heat pump total  : {total_cost_hp:>12.0f} €")
print(f"  Total costs      : {total_cost:>12.0f} €")

print(f"\n── Total annual costs (CAPEX + OPEX) ──────")
print(f"  Total            : {(total_annualised + total_cost)/1e3:>12.1f} k€/a")

h_boiler_only = int(np.sum((P_boiler_res > 0.01) & (P_hp_res  <= 0.01)))
h_hp_only     = int(np.sum((P_hp_res  > 0.01) & (P_boiler_res <= 0.01)))
h_both        = int(np.sum((P_boiler_res > 0.01) & (P_hp_res   > 0.01)))

print(f"\n── Operating hours ────────────────────────")
print(f"  Boiler only      : {h_boiler_only} h")
print(f"  Heat pump only   : {h_hp_only} h")
print(f"  Both active      : {h_both} h")

# ─────────────────────────────────────────────
# 8. Visualization
# ─────────────────────────────────────────────
t_axis = np.arange(N)

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle("Dispatch Optimization: Gas Boiler + Heat Pump\n"
             "(Electricity DE 2024 | Gas DE 2017 | Demand Flensburg 2017 ×0.026)",
             fontsize=12, fontweight="bold")

# Plot 1: Plant dispatch
ax1 = axes[0]
ax1.stackplot(t_axis, P_boiler_res, P_hp_res,
              labels=["Gas Boiler (η=0.98)", "Heat Pump (COP=3.5)"],
              colors=["#e07b39", "#4a90d9"], alpha=0.85)
ax1.plot(t_axis, demand, "k--", lw=1.0, label="Heat Demand")
ax1.set_ylabel("Thermal Power [MW]")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_title("Plant Dispatch")

# Plot 2: Price signals
ax2 = axes[1]
ax2.plot(t_axis, gas_cost_th, color="#e07b39", lw=1.0, label="Gas (per MWh_th)")
ax2.plot(t_axis, hp_cost_th,  color="#4a90d9", lw=1.0, label="Electricity → Heat (per MWh_th)")
ax2.fill_between(t_axis, gas_cost_th, hp_cost_th,
                 where=(gas_cost_th < hp_cost_th),
                 alpha=0.30, color="#e07b39", label="Gas cheaper")
ax2.axhline(0, color="gray", lw=0.5, ls="--")
ax2.set_ylabel("Cost [€/MWh_th]")
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_title("Effective Heat Generation Costs")

# Plot 3: Operating mode
mode = np.where(P_boiler_res > 0.01, 2, 0) + np.where(P_hp_res > 0.01, 1, 0)
colors_mode = {3: "#8e44ad", 2: "#e07b39", 1: "#4a90d9", 0: "#cccccc"}
bar_colors  = [colors_mode[int(m)] for m in mode]
ax3 = axes[2]
ax3.bar(t_axis, np.ones(N), color=bar_colors, width=1.0)
legend_elements = [
    Patch(facecolor="#e07b39", label="Boiler only"),
    Patch(facecolor="#4a90d9", label="Heat pump only"),
    Patch(facecolor="#8e44ad", label="Both active"),
    Patch(facecolor="#cccccc", label="No plant"),
]
ax3.legend(handles=legend_elements, loc="upper right", fontsize=8)
ax3.set_ylabel("Operating Mode")
ax3.set_xlabel("Hour of year")
ax3.set_yticks([])
ax3.grid(True, alpha=0.3, axis="x")
ax3.set_title("Operating Mode per Hour")

plt.tight_layout()
plt.savefig("dispatch_optimization_result.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# 9. Additional analysis plots
# ─────────────────────────────────────────────
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Per-hour cost vectors
cost_boiler_h = gas_cost_th * P_boiler_res   # €/h
cost_hp_h     = hp_cost_th  * P_hp_res       # €/h

# Aggregate per month
monthly_cost_boiler = np.array([cost_boiler_h[months == m].sum() for m in range(1, 13)])
monthly_cost_hp     = np.array([cost_hp_h[months == m].sum()     for m in range(1, 13)])

# Monthly operating hours per mode
monthly_h_boiler = np.array([
    int(np.sum((P_boiler_res[months == m] > 0.01) & (P_hp_res[months == m] <= 0.01)))
    for m in range(1, 13)])
monthly_h_hp = np.array([
    int(np.sum((P_hp_res[months == m] > 0.01) & (P_boiler_res[months == m] <= 0.01)))
    for m in range(1, 13)])
monthly_h_both = np.array([
    int(np.sum((P_boiler_res[months == m] > 0.01) & (P_hp_res[months == m] > 0.01)))
    for m in range(1, 13)])

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Dispatch Optimization: Monthly & Duration Analysis",
              fontsize=12, fontweight="bold")

x_m = np.arange(12)

# Plot 4: Monthly cost breakdown (OPEX + annualised CAPEX)
monthly_capex = np.full(12, total_annualised / 12)   # equal monthly share of annualised CAPEX
ax4 = axes2[0]
ax4.bar(x_m, monthly_cost_boiler / 1e3, color="#e07b39", alpha=0.85, label="Gas Boiler (OPEX)")
ax4.bar(x_m, monthly_cost_hp / 1e3, bottom=monthly_cost_boiler / 1e3,
        color="#4a90d9", alpha=0.85, label="Heat Pump (OPEX)")
opex_total = monthly_cost_boiler + monthly_cost_hp
ax4.bar(x_m, monthly_capex / 1e3, bottom=opex_total / 1e3,
        color="#2ecc71", alpha=0.85, label=f"CAPEX (ann. {total_annualised/1e3:.1f} k€/a)")
ax4.set_xticks(x_m)
ax4.set_xticklabels(month_labels, fontsize=8)
ax4.set_ylabel("Cost [k€]")
ax4.set_title("Monthly Cost Breakdown (OPEX + CAPEX)")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")

# Plot 5: Monthly operating hours
ax5 = axes2[1]
ax5.bar(x_m, monthly_h_boiler, color="#e07b39", alpha=0.85, label="Boiler only")
ax5.bar(x_m, monthly_h_hp,     bottom=monthly_h_boiler,
        color="#4a90d9", alpha=0.85, label="Heat pump only")
ax5.bar(x_m, monthly_h_both,   bottom=monthly_h_boiler + monthly_h_hp,
        color="#8e44ad", alpha=0.85, label="Both active")
ax5.set_xticks(x_m)
ax5.set_xticklabels(month_labels, fontsize=8)
ax5.set_ylabel("Hours [h]")
ax5.set_title("Monthly Operating Hours by Mode")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")

# Plot 6: Load duration curve
ax6 = axes2[2]
sorted_idx     = np.argsort(demand)[::-1]
demand_sorted  = demand[sorted_idx]
boiler_sorted  = P_boiler_res[sorted_idx]
hp_sorted      = P_hp_res[sorted_idx]
dur_axis       = np.arange(1, N + 1)
ax6.stackplot(dur_axis, boiler_sorted, hp_sorted,
              colors=["#e07b39", "#4a90d9"], alpha=0.85,
              labels=["Gas Boiler", "Heat Pump"])
ax6.plot(dur_axis, demand_sorted, "k--", lw=1.0, label="Demand")
ax6.set_xlabel("Hours per year (sorted)")
ax6.set_ylabel("Thermal Power [MW]")
ax6.set_title("Load Duration Curve")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dispatch_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
