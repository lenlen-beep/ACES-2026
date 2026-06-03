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
- Data: heat demand time series (Excel)
- Optimisation: Pyomo (linear optimisation model)
- Objective: minimisation of electricity and gas costs
- Constraints: heat balance + storage dynamics
"""

load_file = r"C:\Users\marie\ACESp\district_heating_data_Flensburg_2017.xlsx"

# pd.read_excel(file,
#               skiprows: number of rows to skip,
#               header:   row containing column headers,
#               usecols:=[] select specific columns,
#               names=[]: rename columns directly,
#               parse_dates=[]: parse column as datetime,
#               index_col: set index column,
#               na_values=['','']: handling of missing values,
#               nrows: limit number of rows loaded (combined with skiprows),
#   )

df_load = pd.read_excel(load_file, skiprows=1, header=0)

# Overwrite column names
df_load.columns = ['Date', 'Heat demand in MW']

df_load['Date'] = pd.to_datetime(df_load['Date'])

# Time index
T = range(len(df_load))

# Demand array
demand = df_load['Heat demand in MW'].values

# Electricity price (synthetic)
price = {t: 50 + 20*np.sin(t/24) for t in T}

# Heat pump parameters
COP      = 3.5   # Coefficient of Performance
P_wp_max = 150   # Maximum electrical input power of heat pump [MW]

# Storage parameters
storage_cap = 500  # Storage capacity [MWh]
charge_max  = 80   # Maximum charge / discharge power [MW]
eta         = 0.9  # Storage efficiency (charge and discharge losses)

# Gas boiler parameters
eta_gb    = 0.92  # Boiler efficiency (conventional boiler ~0.90–0.93)
P_gb_max  = 100   # Maximum thermal output of gas boiler [MW]
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

# General syntax:
#model.x = pyo.Var(model.T, bounds=(min, max)) -- bounded range
#model.x = pyo.Var(model.T)                    -- unbounded
#model.x = pyo.Var(model.T, bounds=(0, None))  -- non-negative only
#model.x = pyo.Var(model.T, bounds=(None, max))-- upper bound only
#model.x = pyo.Var(model.T, within=pyo.Integers) -- integer
#model.x = pyo.Var(model.T, within=pyo.Binary)   -- binary (on/off)

model.P_wp      = pyo.Var(model.T, bounds=(0, P_wp_max))   # Electrical input power of heat pump [MW]
model.charge    = pyo.Var(model.T, bounds=(0, charge_max)) # Storage charging power [MW]
model.discharge = pyo.Var(model.T, bounds=(0, charge_max)) # Storage discharging power [MW]
model.SOC       = pyo.Var(model.T, bounds=(0, storage_cap))# State of charge [MWh]
model.Q_gb      = pyo.Var(model.T, bounds=(0, P_gb_max))   # Thermal output of gas boiler [MW]


# Objective function

# General syntax:
#def name(model):
#    return ...
#model.obj = pyo.Objective(rule=name, sense=pyo.minimize)
#                                          =pyo.maximize)

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

# General syntax:
# Rules that must be satisfied at every time step.
# Always:
#       Equality:   A == B
#       Inequality: A <= B, A >= B

#def rule_name(m, t):
#    return ...
#model.constraint_name = pyo.Constraint(model.T, rule=rule_name)

# Heat balance
# Available heat (COP * P_wp + storage discharge + gas boiler)
#   == Required heat (demand + storage charge)
# The solver automatically dispatches the gas boiler only when the heat pump
# and storage cannot cover demand alone (peak load behaviour).
def heat_balance(m, t):
    return COP * m.P_wp[t] + m.discharge[t] + m.Q_gb[t] == demand[t] + m.charge[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Storage dynamics
def storage_rule(m, t):
    if t == 0:
        # Start with empty storage
        return m.SOC[t] == 0
    # SOC equals SOC from previous time step plus charging minus discharging
    # eta * charge:       charging losses (less energy stored than injected)
    # (1/eta) * discharge: discharging losses (more must be drawn than delivered)
    return m.SOC[t] == m.SOC[t-1] + eta*m.charge[t] - (1/eta)*m.discharge[t]

model.storage = pyo.Constraint(model.T, rule=storage_rule)

# Solver
solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

# Results
P_wp_res      = np.array([pyo.value(model.P_wp[t])      for t in T])
charge_res    = np.array([pyo.value(model.charge[t])    for t in T])
discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
Q_gb_res      = np.array([pyo.value(model.Q_gb[t])      for t in T])
demand        = np.array(demand)

# Key performance indicators
print(f"Heat pump  – Total heat output: {(COP * P_wp_res).sum():.1f} MWh")
print(f"Gas boiler – Total heat output: {Q_gb_res.sum():.1f} MWh")
print(f"Gas boiler – Full load hours:   {(Q_gb_res > 0.01 * P_gb_max).sum()} h")

# Sort by descending demand for duration curves
sorted_idx = np.argsort(-demand)

demand_sorted    = demand[sorted_idx]
wp_sorted        = (COP * P_wp_res)[sorted_idx]
discharge_sorted = discharge_res[sorted_idx]
gb_sorted        = Q_gb_res[sorted_idx]

# Plots

# Duration curve: demand, heat pump, storage discharge, gas boiler
plt.figure()
plt.plot(demand_sorted,    label="Heat demand")
plt.plot(wp_sorted,        label="Heat pump")
plt.plot(discharge_sorted, label="Storage discharge")
plt.plot(gb_sorted,        label="Gas boiler")
plt.xlabel("Hours (sorted)")
plt.ylabel("Power [MW]")
plt.title("Duration curve: heat pump, storage and gas boiler dispatch")
plt.legend()
plt.grid()
plt.show()

# Storage charging and discharging cycles
plt.figure()
plt.plot(charge_res,    label="Charging")
plt.plot(discharge_res, label="Discharging")
plt.xlabel("Time [h]")
plt.ylabel("Power [MW]")
plt.title("Storage charging and discharging cycles")
plt.legend()
plt.grid()
plt.show()

# Time series: all generators
plt.figure()
plt.plot(COP * P_wp_res, label="Heat pump")
plt.plot(discharge_res,  label="Storage discharge")
plt.plot(Q_gb_res,       label="Gas boiler")
plt.plot(demand,         label="Heat demand")
plt.xlabel("Time [h]")
plt.ylabel("Power [MW]")
plt.title("Time series: heat pump, storage and gas boiler dispatch")
plt.legend()
plt.grid()
plt.show()