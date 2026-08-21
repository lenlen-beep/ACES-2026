from funcs.paths import PARAMETERS_FILE
import pyomo.environ as pyo
import numpy as np

from funcs.read_data import read_parameters


"""
Optimization of a district heating system with heat pump and thermal buffer storage.

The model minimizes electricity and storage costs while satisfying the heat demand.
The heat pump can provide heat directly to cover the load or flexibly charge a storage.

Storage dynamics are described via the State of Charge (SOC), which changes each
timestep through charging, discharging and thermal losses:

SOC(t) = SOC(t-1) + charge(t) - discharge(t) - losses

The maximum storable energy is derived from storage volume, water properties,
and the temperature difference between supply and return.

Methodology:
- Time-series based heat demand data
- Linear optimization model using Pyomo
- Heat balance and storage dynamics as constraints
- Cost minimization for heat pump and storage investment
"""

parameters = read_parameters(PARAMETERS_FILE)

# Network parameters
cp_w = parameters["net_parameters"]["specific_heat_capacity"]
rho_w = parameters["net_parameters"]["density"]
s_temp = parameters["net_parameters"]["supply_temperature"]
delta_T = parameters["net_parameters"]["delta_T"]

# Investment parameters
r = parameters["invest_parameters"]["interest_rate"]
n = parameters["invest_parameters"]["lifetime_years"]

storage_invest_offset = parameters["system_parameters"]["storage"]["invest_offset_storage"]
storage_specific_cost = parameters["system_parameters"]["storage"]["specific_invest_storage"]

hp_invest_offset = parameters["system_parameters"]["HP"]["invest_offset_hp"]
hp_specific_cost = parameters["system_parameters"]["HP"]["specific_invest_hp"]

gas_invest_offset = parameters["system_parameters"]["gas_boiler"]["invest_offset_boiler"]
gas_specific_cost = parameters["system_parameters"]["gas_boiler"]["specific_invest_gas_boiler"]

pv_invest_offset = parameters["system_parameters"]["PV"]["invest_offset_pv"]
pv_specific_cost = parameters["system_parameters"]["PV"]["specific_invest_pv"]

seasonal_invest_offset = parameters["system_parameters"]["seasonal_storage"]["invest_offset_seasonal_storage"]
seasonal_specific_cost = parameters["system_parameters"]["seasonal_storage"]["specific_invest_seasonal_storage"]

annuity_factor = r * (1+r)**n / ((1+r)**n - 1)

# Pump parameters
g = parameters["system_parameters"]["pump"]["gravity"]
h = parameters["system_parameters"]["pump"]["del_height"]
eta_pump = parameters["system_parameters"]["pump"]["eta_pump"]

P_pump = (g * h) / (eta_pump * cp_w*1000 *delta_T) #MW

# Heat pump parameters
hp_cap = parameters["system_parameters"]["HP"]["initial_hp_capacity"]
COP = parameters["system_parameters"]["HP"]["COP"]

# Buffer storage parameters
max_charge_rate = parameters["system_parameters"]["storage"]["max_charge_rate"]
max_discharge_rate = parameters["system_parameters"]["storage"]["max_discharge_rate"]
Q_loss = parameters["system_parameters"]["storage"]["Q_loss"]
SOC_init = parameters["system_parameters"]["storage"]["SOC_init"]
storage_cap = parameters["system_parameters"]["storage"]["initial_storage_capacity"]

# Gas boiler parameters
eta_gas_boiler = parameters["system_parameters"]["gas_boiler"]["eta_gas_boiler"]
gas_boiler_cap = parameters["system_parameters"]["gas_boiler"]["initial_gas_thermal_power"]

# PV parameters
pv_cap = parameters["system_parameters"]["PV"]["initial_pv_capacity"]
feed_in_tariff = parameters["system_parameters"]["PV"]["feed_in_tariff"]

# Seasonal storage parameters
seasonal_cap = parameters["system_parameters"]["seasonal_storage"]["initial_seasonal_storage_capacity"]
seasonal_charge_rate = parameters["system_parameters"]["seasonal_storage"]["max_seasonal_charge_rate"]
seasonal_discharge_rate = parameters["system_parameters"]["seasonal_storage"]["max_seasonal_discharge_rate"]


def storage_volume_to_MWh(vol_m3):
    return vol_m3 * rho_w * cp_w * delta_T / 3_600_000.0  # MWh


def optimize_energy_system(
    demand,
    electricity_price,
    gas_price,
    pv,
    cop=None,
    elec_price_mode: str = "spot",
    elec_hedge_share: float = 0.0,
    gas_price_mode: str = "spot",
):
    """
    elec_price_mode  : "spot"   – spot price time series (electricity_price)
                       "tariff" – fixed tariff from parameters.yaml (price_parameters.electricity.tarif.usual_mid)
                       "hedge"  – mix: elec_hedge_share * tariff + (1-elec_hedge_share) * spot
    elec_hedge_share : fixed-tariff share for mode="hedge", e.g. 0.3 = 30 % tariff, 70 % spot
    gas_price_mode   : "spot"   – spot price time series (gas_price)
                       "tariff" – fixed tariff from parameters.yaml (price_parameters.gas.tarif.usual_mid)
    cop              : optional, array-like (same length as demand) – time-varying COP of the heat pump,
                        e.g. COP_t from era5_weather.compute_cop().
                        If None (default), the static COP from parameters.yaml
                        (system_parameters.HP.COP) is used.
    """
    # Wholesale spot price → all-in end-customer price, comparable to the all-in gas tariff.
    # Must be applied before the elec_price_mode logic: usual_mid is already all-in.
    electricity_price = (np.asarray(electricity_price, dtype=float)
                         + elec_volumetric_surcharge(parameters))

    demand = demand.values
    T = range(len(demand))

    if cop is None:
        cop_t = np.full(len(T), COP)
    else:
        cop_t = np.asarray(getattr(cop, "values", cop), dtype=float)

    # Electricity price preparation (unit: €/MWh)
    elec_tariff_eur_per_mwh = parameters["price_parameters"]["electricity"]["tarif"]["usual_mid"] * 10  # ct/kWh → €/MWh
    if elec_price_mode == "tariff":
        electricity_price = np.full(len(T), elec_tariff_eur_per_mwh)
    elif elec_price_mode == "hedge":
        electricity_price = elec_hedge_share * elec_tariff_eur_per_mwh + (1 - elec_hedge_share) * electricity_price

    # Gas price preparation (unit: €/MWh)
    gas_tariff_eur_per_mwh = parameters["price_parameters"]["gas"]["tarif"]["usual_mid"] * 10  # ct/kWh → €/MWh
    if gas_price_mode == "tariff":
        gas_price = np.full(len(T), gas_tariff_eur_per_mwh)

    # Warning if supply temperature is too high (buffer storage unsuitable)
    if s_temp > 95:
        import warnings
        warnings.warn("Supply temp > 95°C: buffer storages are not suitable. Please check temperature limits.", UserWarning)

    # Model
    model = pyo.ConcreteModel()

    # Time index
    model.T = pyo.Set(initialize=T)


    # --------------------------------
    # Variables
    # --------------------------------

    # Heat pump
    model.hp_capacity = pyo.Var(bounds=(0, None), initialize=hp_cap) #MW
    model.Q_hp = pyo.Var(model.T, bounds=(0, None))
    model.P_el_hp = pyo.Var(model.T, bounds=(0, None))

    # Buffer storage
    model.charge = pyo.Var(model.T, bounds=(0, None)) #MW
    model.discharge = pyo.Var(model.T, bounds=(0, None)) #MW
    model.SOC = pyo.Var(model.T, bounds=(0, None)) #MWh
    model.storage_capacity = pyo.Var(bounds=(0, None), initialize=storage_cap) #m3

    # Gas boiler
    model.Q_gas_boiler = pyo.Var(model.T, bounds=(0, None)) #MW
    model.gas_boiler_capacity = pyo.Var(bounds=(0, None), initialize=gas_boiler_cap) #MW

    # PV
    model.P_grid = pyo.Var(model.T, bounds=(0, None)) #MW
    model.pv_feed_in = pyo.Var(model.T, bounds=(0, None)) #MW
    model.pv_availability = pyo.Var(model.T, bounds=(0, None)) #MW
    model.pv_capacity = pyo.Var(bounds=(0, None), initialize=pv_cap) #MW

    # Seasonal storage
    model.seasonal_charge    = pyo.Var(model.T, bounds=(0, None))  # MW
    model.seasonal_discharge = pyo.Var(model.T, bounds=(0, None))  # MW
    model.SOC_seasonal       = pyo.Var(model.T, bounds=(0, None))  # MWh
    model.seasonal_capacity  = pyo.Var(bounds=(0, None), initialize=seasonal_cap)  # m³


    # --------------------------------
    # Objective function
    # --------------------------------

    def obj_rule(m):
        return sum(electricity_price[t] * m.P_grid[t] for t in m.T) + \
                (storage_invest_offset + storage_specific_cost * m.storage_capacity) * annuity_factor + \
                (hp_invest_offset + hp_specific_cost * m.hp_capacity) * annuity_factor + \
                P_pump * sum(electricity_price[t] * (m.charge[t] + m.discharge[t]) for t in m.T) + \
                (gas_invest_offset + gas_specific_cost * m.gas_boiler_capacity) * annuity_factor + \
                sum(gas_price[t] * (m.Q_gas_boiler[t] / eta_gas_boiler) for t in m.T) + \
                (pv_invest_offset + pv_specific_cost * m.pv_capacity) * annuity_factor - \
                sum(feed_in_tariff * m.pv_feed_in[t] for t in m.T) + \
                (seasonal_invest_offset + seasonal_specific_cost * m.seasonal_capacity) * annuity_factor

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


    # --------------------------------
    # Constraints
    # --------------------------------

    # Heat balance
    def heat_balance(m, t):
        return m.Q_hp[t] + m.Q_gas_boiler[t] + m.discharge[t] + m.seasonal_discharge[t] == demand[t] + m.charge[t] + m.seasonal_charge[t]

    model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

    # Heat pump COP (static or time-varying, see cop parameter above)
    def cop_rule(m, t):
        return m.Q_hp[t] == cop_t[t] * m.P_el_hp[t]

    model.cop_constraint = pyo.Constraint(model.T, rule=cop_rule)

    # Heat pump capacity limit
    def hp_capacity_rule(m, t):
        return m.Q_hp[t] <= m.hp_capacity

    model.hp_capacity_constraint = pyo.Constraint(model.T, rule=hp_capacity_rule)


    # Gas boiler capacity limit
    def gas_boiler_capacity_rule(m, t):
        return m.Q_gas_boiler[t] <= m.gas_boiler_capacity

    model.gas_boiler_capacity_constraint = pyo.Constraint(model.T, rule=gas_boiler_capacity_rule)

    # Gas share of annual heat demand: max. 10 %
    def gas_boiler_share_rule(m):
        return sum(m.Q_gas_boiler[t] for t in m.T) <= 0.1 * sum(demand)

    model.gas_boiler_share_constraint = pyo.Constraint(rule=gas_boiler_share_rule)


    # Buffer storage dynamics
    def storage_rule(m, t):
        # Initial and final SOC are fixed to a predefined fraction to ensure realistic
        # operating conditions (avoids permanently empty or full storage scenarios).
        # All other timesteps: SOC = previous SOC + charge - discharge - losses.
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        Q_loss_MWh = Q_loss

        if t == 0:
            return m.SOC[t] == SOC_init * storage_MWh  # initial SOC (start)

        elif t == T[-1]:
            return m.SOC[t] == SOC_init * storage_MWh  # initial SOC (end)

        return m.SOC[t] == m.SOC[t-1] + m.charge[t] - m.discharge[t] - Q_loss_MWh

    model.storage = pyo.Constraint(model.T, rule=storage_rule)

    # Storage capacity limit
    def soc_capacity_limit(m, t):
        # SOC cannot exceed the storage capacity
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        return m.SOC[t] <= storage_MWh

    model.soc_capacity_limit = pyo.Constraint(model.T, rule=soc_capacity_limit)

    # Charging power limit (storage)
    def charge_power_limit_storage(m, t):
        # Charging power is limited by the maximum charge rate and storage capacity
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        return m.charge[t] <= max_charge_rate * storage_MWh

    model.charge_power_limit_storage = pyo.Constraint(model.T, rule=charge_power_limit_storage)

    # Charging power limit (heat pump)
    def charge_power_limit_hp(m, t):
        # Charging power is limited by the heat pump capacity,
        # since storage can only be charged via the heat pump
        return m.charge[t] <= m.hp_capacity

    model.charge_power_limit_hp = pyo.Constraint(model.T, rule=charge_power_limit_hp)

    # Discharging power limit
    def discharge_power_limit(m, t):
        # Discharging power is limited by the maximum discharge rate and storage capacity
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        return m.discharge[t] <= max_discharge_rate * storage_MWh

    model.discharge_power_limit = pyo.Constraint(model.T, rule=discharge_power_limit)

    # Discharge restriction during negative electricity prices
    def negative_price_discharge_restrict(m, t):
        # When the electricity price is negative, discharging is set to zero
        # to prevent the storage from discharging during periods of negative prices
        # (economically counterproductive and causes artificial cycling).
        if electricity_price[t] < 0:
            return m.discharge[t] == 0
        else:
            return pyo.Constraint.Skip

    model.negative_price_discharge_restrict = pyo.Constraint(model.T, rule=negative_price_discharge_restrict)


    # PV availability
    def pv_availability_rule(m, t):
        return m.pv_availability[t] == pv[t] * m.pv_capacity
    model.pv_availability_constraint = pyo.Constraint(model.T, rule=pv_availability_rule)

    # Electricity balance: grid + PV = heat pump + feed-in
    def electricity_balance(m, t):
        return m.P_grid[t] + m.pv_availability[t] == m.P_el_hp[t] + m.pv_feed_in[t]
    model.elec_balance = pyo.Constraint(model.T, rule=electricity_balance)

    # PV feed-in limit
    def feed_in_limit(m, t):
        return m.pv_feed_in[t] <= m.pv_availability[t]
    model.feed_in_limit = pyo.Constraint(model.T, rule=feed_in_limit)


    def seasonal_storage_rule(m, t):
        seasonal_MWh = storage_volume_to_MWh(m.seasonal_capacity)
        initial_soc  = 0.5 * seasonal_MWh
        if t == 0:
            return m.SOC_seasonal[t] == initial_soc + m.seasonal_charge[t] - m.seasonal_discharge[t]
        return m.SOC_seasonal[t] == (m.SOC_seasonal[t-1]
                                    + m.seasonal_charge[t]
                                    - m.seasonal_discharge[t])
    model.seasonal_storage = pyo.Constraint(model.T, rule=seasonal_storage_rule)

    def seasonal_soc_limit(m, t):
        return m.SOC_seasonal[t] <= storage_volume_to_MWh(m.seasonal_capacity)
    model.seasonal_soc_limit = pyo.Constraint(model.T, rule=seasonal_soc_limit)

    SUMMER = set(range(2880, 6552))  # April–September (hours 2880–6552)

    def _rule_seasonal_charge_summer(m, t):
        if t not in SUMMER:
            return m.seasonal_charge[t] == 0
        return pyo.Constraint.Skip
    model.seasonal_charge_only_summer = pyo.Constraint(model.T, rule=_rule_seasonal_charge_summer)

    def _rule_seasonal_discharge_winter(m, t):
        if t in SUMMER:
            return m.seasonal_discharge[t] == 0
        return pyo.Constraint.Skip
    model.seasonal_discharge_only_winter = pyo.Constraint(model.T, rule=_rule_seasonal_discharge_winter)

    def seasonal_soc_end(m):
        seasonal_MWh = storage_volume_to_MWh(m.seasonal_capacity)
        return m.SOC_seasonal[T[-1]] == 0.5 * seasonal_MWh
    model.seasonal_soc_end = pyo.Constraint(rule=seasonal_soc_end)

    def seasonal_charge_limit(m, t):
        return m.seasonal_charge[t] <= seasonal_charge_rate * storage_volume_to_MWh(m.seasonal_capacity)
    model.seasonal_charge_limit = pyo.Constraint(model.T, rule=seasonal_charge_limit)

    def seasonal_discharge_limit(m, t):
        return m.seasonal_discharge[t] <= seasonal_discharge_rate * storage_volume_to_MWh(m.seasonal_capacity)
    model.seasonal_discharge_limit = pyo.Constraint(model.T, rule=seasonal_discharge_limit)


    # Solver
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model, tee=True)

    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not converge: {results.solver.termination_condition}")

# --------------------------------------------------
    # Results
# --------------------------------------------------
    charge_res = np.array([pyo.value(model.charge[t]) for t in T])
    discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
    SOC_res = np.array([pyo.value(model.SOC[t]) for t in T])
    storage_cap_res = pyo.value(model.storage_capacity)
    hp_capacity_res = pyo.value(model.hp_capacity)
    Q_hp_res = np.array([pyo.value(model.Q_hp[t]) for t in T])
    Q_gas_boiler_res = np.array([pyo.value(model.Q_gas_boiler[t]) for t in T])
    gas_boiler_cap_res = pyo.value(model.gas_boiler_capacity)
    pv_res = np.array([pyo.value(model.pv_availability[t]) for t in T])
    pv_feed_in_res = np.array([pyo.value(model.pv_feed_in[t]) for t in T])
    pv_cap_res = pyo.value(model.pv_capacity)
    seasonal_charge_res    = np.array([pyo.value(model.seasonal_charge[t]) for t in T])
    seasonal_discharge_res = np.array([pyo.value(model.seasonal_discharge[t]) for t in T])
    seasonal_soc_res       = np.array([pyo.value(model.SOC_seasonal[t]) for t in T])
    seasonal_cap_res       = pyo.value(model.seasonal_capacity)

    print(f'Buffer storage size: {storage_cap_res:.1f} m³  ({storage_volume_to_MWh(storage_cap_res):.1f} MWh)')
    print(f'Heat pump capacity: {hp_capacity_res:.1f} MW')
    print(f'Gas boiler capacity: {gas_boiler_cap_res:.1f} MW')
    print(f'Gas share of annual heat: {Q_gas_boiler_res.sum() / sum(demand) * 100:.1f} %')
    print(f'PV capacity: {pv_cap_res:.1f} MW')
    print(f'PV feed-in: {pv_feed_in_res.sum():.1f} MWh  (revenue: {pv_feed_in_res.sum() * feed_in_tariff:.1f} €)')
    print(f'PV availability: {pv_res.sum():.1f} MWh')
    seasonal_cap_MWh = storage_volume_to_MWh(seasonal_cap_res)
    print(f'Seasonal storage capacity:  {seasonal_cap_res:.1f} m³ = {seasonal_cap_MWh:.2f} MWh')
    print(f'Seasonal storage peak SOC:  {seasonal_soc_res.max():.2f} MWh  ({seasonal_soc_res.max()/seasonal_cap_MWh*100:.0f}% of capacity)' if seasonal_cap_MWh > 0 else '')
    print(f'Seasonal storage charge:    {seasonal_charge_res.sum():.2f} MWh/a')
    print(f'Seasonal storage discharge: {seasonal_discharge_res.sum():.2f} MWh/a')

    return (results, Q_hp_res, Q_gas_boiler_res, charge_res, discharge_res, SOC_res,
            storage_cap_res, gas_boiler_cap_res, hp_capacity_res,
            pv_res, pv_feed_in_res, pv_cap_res,
            seasonal_charge_res, seasonal_discharge_res, seasonal_soc_res, seasonal_cap_res)


def elec_volumetric_surcharge(parameters) -> float:
    """Volumetric surcharges on the spot price in €/MWh.

    Includes network commodity charge, electricity tax, network-fee-based levies
    and concession fee. The capacity charge is NOT included — it is not
    marginal-cost-relevant and enters the LCOH as a fixed-cost term.
    """
    el  = parameters["price_parameters"]["electricity"]
    vbh = el.get("vbh_class", "lower_2500VBH")
    ct_per_kwh = (
        el["network_charge"][vbh]["commodity_charge"]
        + el["tax"]
        + sum(el["levies"].values())
        + el["concession_fee"]
    )
    return ct_per_kwh * 10.0   # ct/kWh -> €/MWh
