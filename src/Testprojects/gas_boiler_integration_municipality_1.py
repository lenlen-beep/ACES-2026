import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt

"""
Optimisation of a district heating system with heat pump, gas boiler and thermal storage.

The model minimises the combined electricity and gas costs while ensuring
heat supply at every time step. The storage enables temporal shifting of
heat production. The gas boiler acts as a peak load generator and backup
when the heat pump and storage cannot meet demand on their own.

Methodology:
- Data: heat demand time series derived from Flensburg 2017 network data
        (Stadtwerke Flensburg / ZNES, DOI 10.5281/zenodo.10508280),
        scaled by factor 0.026 to simulate a ~2,000 pop. rural municipality
        (~30 GWh/a, peak ~9 MW)
- Optimisation: Pyomo (linear optimisation model)
- Objective: minimisation of electricity and gas costs
- Constraints: heat balance + storage dynamics
"""

load_file = r"Flensburg2017_Municipality_scale026.xlsx"

df_load = pd.read_excel(
    load_file,
    sheet_name="Hourly Data",
    usecols=["Timestamp", "Q_Municipality (MW)"],
)

df_load.columns = ["Date", "Heat demand in MW"]
df_load["Date"] = pd.to_datetime(df_load["Date"])

# Time index
T = range(len(df_load))

# Demand array
demand = df_load["Heat demand in MW"].values

# Electricity price (synthetic)
price = {t: 50 + 20 * np.sin(t / 24) for t in T}

# Heat pump parameters
# 1.5 MW electrical covers ~58 % of peak as base-load heat pump (typical for small DH)
COP      = 3.5   # Coefficient of Performance
P_wp_max = 1.5   # Maximum electrical input power of heat pump [MW]

# Storage parameters
# ~10 MWh ≈ 1 h of average load; 2 MW charge/discharge is ~22 % of peak (typical buffer)
storage_cap = 10   # Storage capacity [MWh]
charge_max  = 2.0  # Maximum charge / discharge power [MW]
eta         = 0.9  # Storage efficiency (charge and discharge losses)

# Gas boiler parameters
# Peak demand ~9 MW; HP covers at most COP * P_wp_max = 5.25 MW + 2 MW storage = 7.25 MW
# → boiler handles remaining peaks up to ~9 MW; 10 MW gives a small safety margin
eta_gb    = 0.92  # Boiler efficiency (conventional boiler ~0.90–0.93)
P_gb_max  = 10    # Maximum thermal output of gas boiler [MW]
gas_price = 30    # Gas price [€/MWh_th fuel input]
#
# Note on gas costs:
# The gas price refers to fuel input energy.
# To produce 1 MWh of heat, the boiler requires 1/eta_gb MWh of gas.
# Cost per MWh of heat = gas_price / eta_gb
# Example: 30 / 0.92 ≈ 32.61 €/MWh_th


# Model
model = pyo.ConcreteModel()

# Time index in model
model.T = pyo.Set(initialize=T)


# Variables

model.P_wp      = pyo.Var(model.T, bounds=(0, P_wp_max))    # Electrical input power of heat pump [MW]
model.charge    = pyo.Var(model.T, bounds=(0, charge_max))  # Storage charging power [MW]
model.discharge = pyo.Var(model.T, bounds=(0, charge_max))  # Storage discharging power [MW]
model.SOC       = pyo.Var(model.T, bounds=(0, storage_cap)) # State of charge [MWh]
model.Q_gb      = pyo.Var(model.T, bounds=(0, P_gb_max))    # Thermal output of gas boiler [MW]


# Objective function

# Minimise total electricity and gas costs:
# - Electricity costs: electricity price [t] * electrical input power of heat pump [t]
# - Gas costs:         (gas price / boiler efficiency) * thermal output of gas boiler [t]
#   --> (gas_price / eta_gb) represents the fuel cost per MWh of useful heat
def obj_rule(m):
    electricity_costs = sum(price[t] * m.P_wp[t] for t in m.T)
    gas_costs         = sum((gas_price / eta_gb) * m.Q_gb[t] for t in m.T)
    return electricity_costs + gas_costs

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


# Constraints

# Heat balance
# Available heat (COP * P_wp + storage discharge + gas boiler)
#   == Required heat (demand + storage charge)
def heat_balance(m, t):
    return COP * m.P_wp[t] + m.discharge[t] + m.Q_gb[t] == demand[t] + m.charge[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Storage dynamics
def storage_rule(m, t):
    if t == 0:
        return m.SOC[t] == 0
    return m.SOC[t] == m.SOC[t - 1] + eta * m.charge[t] - (1 / eta) * m.discharge[t]

model.storage = pyo.Constraint(model.T, rule=storage_rule)


# Solver
solver  = pyo.SolverFactory("glpk")
results = solver.solve(model)

status = results.solver.termination_condition
if status != pyo.TerminationCondition.optimal:
    max_demand  = demand.max()
    min_gb_need = max_demand - COP * P_wp_max - charge_max
    raise RuntimeError(
        f"Solver status: {status}. "
        f"Peak demand is {max_demand:.4f} MW; "
        f"P_gb_max must be at least {min_gb_need:.4f} MW."
    )


# Results
P_wp_res      = np.array([pyo.value(model.P_wp[t])      for t in T])
charge_res    = np.array([pyo.value(model.charge[t])    for t in T])
discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
Q_gb_res      = np.array([pyo.value(model.Q_gb[t])      for t in T])
demand        = np.array(demand)

# Key performance indicators
total_demand   = demand.sum()
hp_heat        = (COP * P_wp_res).sum()
gb_heat        = Q_gb_res.sum()
gb_hours       = (Q_gb_res > 0.01 * P_gb_max).sum()
hp_share_pct   = hp_heat / total_demand * 100
gb_share_pct   = gb_heat / total_demand * 100

print(f"Annual heat demand        : {total_demand:.1f} MWh  ({total_demand/1000:.2f} GWh)")
print(f"Heat pump  – heat output  : {hp_heat:.1f} MWh  ({hp_share_pct:.1f} %)")
print(f"Gas boiler – heat output  : {gb_heat:.1f} MWh  ({gb_share_pct:.1f} %)")
print(f"Gas boiler – full-load h  : {gb_hours} h")
print(f"Peak demand               : {demand.max():.4f} MW")
print(f"Peak gas boiler output    : {Q_gb_res.max():.4f} MW")


# Sort by descending demand for duration curves
sorted_idx = np.argsort(-demand)

demand_sorted    = demand[sorted_idx]
wp_sorted        = (COP * P_wp_res)[sorted_idx]
discharge_sorted = discharge_res[sorted_idx]
gb_sorted        = Q_gb_res[sorted_idx]


# Plots

# Duration curve: demand, heat pump, storage discharge, gas boiler
plt.figure(figsize=(10, 5))
plt.plot(demand_sorted,    label="Heat demand")
plt.plot(wp_sorted,        label="Heat pump")
plt.plot(discharge_sorted, label="Storage discharge")
plt.plot(gb_sorted,        label="Gas boiler")
plt.xlabel("Hours (sorted by descending demand)")
plt.ylabel("Power [MW]")
plt.title("Duration curve – heat pump, storage and gas boiler dispatch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Storage charging and discharging cycles
plt.figure(figsize=(10, 5))
plt.plot(charge_res,    label="Charging")
plt.plot(discharge_res, label="Discharging")
plt.xlabel("Time [h]")
plt.ylabel("Power [MW]")
plt.title("Storage charging and discharging cycles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Time series: all generators
plt.figure(figsize=(10, 5))
plt.plot(COP * P_wp_res, label="Heat pump",         alpha=0.8)
plt.plot(discharge_res,  label="Storage discharge", alpha=0.8)
plt.plot(Q_gb_res,       label="Gas boiler",        alpha=0.8)
plt.plot(demand,         label="Heat demand",       linewidth=1.2, color="black")
plt.xlabel("Time [h]")
plt.ylabel("Power [MW]")
plt.title("Time series – heat pump, storage and gas boiler dispatch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
