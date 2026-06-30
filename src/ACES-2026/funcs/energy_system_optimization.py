import pyomo.environ as pyo
import numpy as np

from .read_data import read_parameters

    # TODO:Der Speicher kann bei SOC = 0 keine Wärme verlieren (theoretisch aber schon, da 55°C RLT) 
    # --> beachten :)

    # TODO: Gas tariflich 

    # TODO: Strom hedge variabel machen (erstmal nachrangig)


"""
Optimierung eines Fernwärmesystems mit Wärmepumpe und thermischem Pufferspeicher.

Das Modell minimiert die Strom- und Speicherkosten unter Einhaltung der
Wärmeversorgung. Die Wärmepumpe kann Wärme direkt zur Lastdeckung bereitstellen
oder zeitlich flexibel in einen Speicher einspeichern.

Die Speicherdynamik wird über den Ladezustand (State of Charge, SOC) beschrieben.
Der SOC verändert sich in jedem Zeitschritt durch Be- und Entladung sowie
thermische Speicherverluste:

SOC(t) = SOC(t-1) + charge(t) - discharge(t) - losses

Die maximal speicherbare Energiemenge ergibt sich aus Speichervolumen,
Wassereigenschaften und Temperaturhub zwischen Vor- und Rücklauf.

Methodik:
- Zeitreihenbasierte Wärmebedarfsdaten
- Lineares Optimierungsmodell mit Pyomo
- Wärmebilanz und Speicherdynamik als Nebenbedingungen
- Kostenminimierung der Wärmepumpe und Speicherinvestition
"""

parameters = read_parameters("src/ACES-2026/parameters.yaml")

#Netzparameter
cp_w = parameters["net_parameters"]["specific_heat_capacity"]
rho_w = parameters["net_parameters"]["density"]
s_temp = parameters["net_parameters"]["supply_temperature"]
delta_T = parameters["net_parameters"]["delta_T"]

#Investitionsparameter
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

# Pumpenparameter
g = parameters["system_parameters"]["pump"]["gravity"]
h = parameters["system_parameters"]["pump"]["del_height"]
eta_pump = parameters["system_parameters"]["pump"]["eta_pump"]

P_pump = (g * h) / (eta_pump * cp_w*1000 *delta_T) #MW

# Wärmepumpenparameter
hp_cap = parameters["system_parameters"]["HP"]["initial_hp_capacity"]
COP = parameters["system_parameters"]["HP"]["COP"]

# Speicherparameter
max_charge_rate = parameters["system_parameters"]["storage"]["max_charge_rate"]
max_discharge_rate = parameters["system_parameters"]["storage"]["max_discharge_rate"]
Q_loss = parameters["system_parameters"]["storage"]["Q_loss"]
SOC_init = parameters["system_parameters"]["storage"]["SOC_init"]
storage_cap = parameters["system_parameters"]["storage"]["initial_storage_capacity"]
Q_loss_seasonal = parameters["system_parameters"]["seasonal_storage"]["Q_loss_seasonal"]

# Gaskesselparameter
eta_gas_boiler = parameters["system_parameters"]["gas_boiler"]["eta_gas_boiler"]
gas_boiler_cap = parameters["system_parameters"]["gas_boiler"]["initial_gas_thermal_power"]

# PV-Parameter
pv_cap = parameters["system_parameters"]["PV"]["initial_pv_capacity"]
feed_in_tariff = parameters["system_parameters"]["PV"]["feed_in_tariff"]

# Saisonaler Speicherparameter
seasonal_cap = parameters["system_parameters"]["seasonal_storage"]["initial_seasonal_storage_capacity"]
seasonal_charge_rate = parameters["system_parameters"]["seasonal_storage"]["max_seasonal_charge_rate"]
seasonal_discharge_rate = parameters["system_parameters"]["seasonal_storage"]["max_seasonal_discharge_rate"]


def storage_volume_to_MWh(vol_m3):
    # vol_m3 [m³] * rho_w [kg/m³] * cp_w [kJ/(kg·K)] * delta_T [K] = kJ
    # kJ / 3600 = kWh,  kWh / 1000 = MWh  →  /3_600_000
    return vol_m3 * rho_w * cp_w * delta_T / 3_600_000.0  # MWh


def optimize_energy_system(
    demand,
    electricity_price,
    gas_price,
    pv,
    elec_price_mode: str = "spot",
    elec_hedge_share: float = 0.0,
    gas_price_mode: str = "spot",
):
    """
    elec_price_mode  : "spot"   – Spotpreisreihe (electricity_price)
                       "tariff" – Festpreis aus parameters.yaml (price_parameters.electricity.tarif.usual_mid)
                       "hedge"  – Mischung: elec_hedge_share * Tarif + (1-elec_hedge_share) * Spot
    elec_hedge_share : Anteil Festpreis bei mode="hedge", z.B. 0.3 = 30 % Tarif, 70 % Spot
    gas_price_mode   : "spot"   – Spotpreisreihe (gas_price)
                       "tariff" – Festpreis aus parameters.yaml (price_parameters.gas.tarif.usual_mid)
    """
    demand = demand.values
    T = range(len(demand))

    # Strompreis aufbereiten (Einheit: €/MWh)
    elec_tariff_eur_per_mwh = parameters["price_parameters"]["electricity"]["tarif"]["usual_mid"] * 10  # ct/kWh → €/MWh
    if elec_price_mode == "tariff":
        electricity_price = np.full(len(T), elec_tariff_eur_per_mwh)
    elif elec_price_mode == "hedge":
        electricity_price = elec_hedge_share * elec_tariff_eur_per_mwh + (1 - elec_hedge_share) * electricity_price
    # else: "spot" → electricity_price unverändert

    # Gaspreis aufbereiten (Einheit: €/MWh)
    gas_tariff_eur_per_mwh = parameters["price_parameters"]["gas"]["tarif"]["usual_mid"] * 10  # ct/kWh → €/MWh
    if gas_price_mode == "tariff":
        gas_price = np.full(len(T), gas_tariff_eur_per_mwh)
    # else: "spot" → gas_price unverändert

    # Warnung, wenn Vorlauftemperatur zu hoch ist (Speicher ungeeignet)
    if s_temp > 95:
        import warnings
        warnings.warn("VLT > 95°C: Pufferspeicher sind dafür nicht geeignet. Bitte Temperaturgrenzen prüfen.", UserWarning)

    # Modell
    model = pyo.ConcreteModel()

    # Zeitindex im Modell
    model.T = pyo.Set(initialize=T)


    # --------------------------------
    # Variablen
    # --------------------------------

    # Wärmepumpe
    model.hp_capacity = pyo.Var(bounds=(0, None), initialize=hp_cap) #MW
    model.Q_hp = pyo.Var(model.T, bounds=(0, None))
    model.P_el_hp = pyo.Var(model.T, bounds=(0, None))

    # Speicher
    model.charge = pyo.Var(model.T, bounds=(0, None)) #MW
    model.discharge = pyo.Var(model.T, bounds=(0, None)) #MW
    model.SOC = pyo.Var(model.T, bounds=(0, None)) #MWh
    model.storage_capacity = pyo.Var(bounds=(0, None), initialize=storage_cap) #m3

    #Gaskessel
    model.Q_gas_boiler = pyo.Var(model.T, bounds=(0, None)) #MW
    model.gas_boiler_capacity = pyo.Var(bounds=(0, None), initialize=gas_boiler_cap) #MW

    # PV
    model.P_grid = pyo.Var(model.T, bounds=(0, None)) #MW
    model.pv_feed_in = pyo.Var(model.T, bounds=(0, None)) #MW
    model.pv_availability = pyo.Var(model.T, bounds=(0, None)) #MW
    model.pv_capacity = pyo.Var(bounds=(0, None), initialize=pv_cap) #MW

    # Saisonaler Speicher
    model.seasonal_charge    = pyo.Var(model.T, bounds=(0, None))  # MW
    model.seasonal_discharge = pyo.Var(model.T, bounds=(0, None))  # MW
    model.SOC_seasonal       = pyo.Var(model.T, bounds=(0, None))  # MWh
    model.seasonal_capacity  = pyo.Var(bounds=(0, None), initialize=seasonal_cap)  # m³


    # --------------------------------
    # Zielfunktion
    # --------------------------------

    # Stromkosten minimieren
    # P_grid[t] statt P_el_hp[t]: nur der tatsächliche Netzbezug wird berechnet,
    # PV-Eigenverbrauch reduziert P_grid und spart damit den vollen Strompreis
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

    # Wärmebilanz
    def heat_balance(m, t):
        #Verfügbare Wärmeleistung (P_WP + P_SP) == Benötigte Wärmeleistung (Last + P_SP)
        return m.Q_hp[t] + m.Q_gas_boiler[t] + m.discharge[t] + m.seasonal_discharge[t] == demand[t] + m.charge[t] + m.seasonal_charge[t]

    model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

    # COP der Wärmepumpe
    def cop_rule(m, t):
        return m.Q_hp[t] == COP * m.P_el_hp[t]

    model.cop_constraint = pyo.Constraint(model.T, rule=cop_rule)

    # Wärmepumpenleistung
    def hp_capacity_rule(m, t):
        return m.Q_hp[t] <= m.hp_capacity

    model.hp_capacity_constraint = pyo.Constraint(model.T, rule=hp_capacity_rule)


    # Gaskesselleistung
    def gas_boiler_capacity_rule(m, t):
        return m.Q_gas_boiler[t] <= m.gas_boiler_capacity

    model.gas_boiler_capacity_constraint = pyo.Constraint(model.T, rule=gas_boiler_capacity_rule)

    # Gasanteil Wärmebedarfsdeckung: max. 10% der Jahresgesamtlast
    def gas_boiler_share_rule(m):
        return sum(m.Q_gas_boiler[t] for t in m.T) <= 0.1 * sum(demand)

    model.gas_boiler_share_constraint = pyo.Constraint(rule=gas_boiler_share_rule)


    # Speicher-Dynamik
    def storage_rule(m, t):
        # Anfangs- und End-SOC auf x% initialen Ladezustand setzen, um realistische 
        # Betriebsbedingungen zu gewährleisten (z.B. Vermeidung von unrealistischen 
        # Szenarien mit dauerhaft leerem oder vollem Speicher).
        #Restliche Zeitschritte: SOC immer gleich dem SOC aus vorherigen Zeitschritt 
        # + charge oder - discharge - Verluste
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        Q_loss_MWh = Q_loss * storage_MWh  # relativer Verlust: 0,2 %/h der Kapazität

        if t == 0:
            return m.SOC[t] == SOC_init * storage_MWh

        elif t == T[-1]:
            return m.SOC[t] == SOC_init * storage_MWh

        return m.SOC[t] == m.SOC[t-1] + m.charge[t] - m.discharge[t] - Q_loss_MWh

    model.storage = pyo.Constraint(model.T, rule=storage_rule)

    # Speicherkapazität
    def soc_capacity_limit(m, t):
        # Speicher kann nicht mehr Energie speichern als die Kapazität erlaubt
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        return m.SOC[t] <= storage_MWh

    model.soc_capacity_limit = pyo.Constraint(model.T, rule=soc_capacity_limit)

    # Ladekapazität
    def charge_power_limit_storage(m, t):
        # Die Ladungsleistung ist begrenzt durch die maximale Ladungsrate 
        # und die Speicherkapazität
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)

        return m.charge[t] <= max_charge_rate * storage_MWh
    
    model.charge_power_limit_storage = pyo.Constraint(model.T, rule=charge_power_limit_storage)

    # Ladekapazität durch Wärmepumpe
    def charge_power_limit_hp(m, t):
        # Die Ladungsleistung ist begrenzt durch die maximale Leistung der Wärmepumpe, 
        # da die Ladung nur über die WP erfolgen kann
        return m.charge[t] <= m.hp_capacity

    model.charge_power_limit_hp = pyo.Constraint(model.T, rule=charge_power_limit_hp)

    # Entladekapazität
    def discharge_power_limit(m, t):
        # Die Entladungsleistung ist begrenzt durch die maximale Entladungsrate
        # und die Speicherkapazität
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        return m.discharge[t] <= max_discharge_rate * storage_MWh

    model.discharge_power_limit = pyo.Constraint(model.T, rule=discharge_power_limit)
    
    # Entladeverbot bei negativen Strompreisen
    def negative_price_discharge_restrict(m, t):
        # Wenn der Strompreis negativ ist, soll die Entladung des Speichers auf 0 
        # begrenzt werden, um zu verhindern, dass der Speicher in Zeiten negativer 
        # Preise entlädt (was wirtschaftlich nicht sinnvoll wäre und zu Zyklen führt).
        if electricity_price[t] < 0:
            return m.discharge[t] == 0 
        else:
            return pyo.Constraint.Skip
        
    model.negative_price_discharge_restrict = pyo.Constraint(model.T, rule=negative_price_discharge_restrict)

    #PV-Verfügbarkeit
    def pv_availability_rule(m, t):
        return m.pv_availability[t] == pv[t] * m.pv_capacity
    model.pv_availability_constraint = pyo.Constraint(model.T, rule=pv_availability_rule)

    # Elektrische Bilanz: Strom aus Netz + PV = Strom für WP + Einspeisung
    def electricity_balance(m, t):
        return m.P_grid[t] + m.pv_availability[t] == m.P_el_hp[t] + m.pv_feed_in[t]
    model.elec_balance = pyo.Constraint(model.T, rule=electricity_balance)

    # Einspeisebegrenzung PV
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
                                    - m.seasonal_discharge[t]
                                    - Q_loss_seasonal * m.SOC_seasonal[t-1])
    model.seasonal_storage = pyo.Constraint(model.T, rule=seasonal_storage_rule)

    def seasonal_soc_limit(m, t):
        return m.SOC_seasonal[t] <= storage_volume_to_MWh(m.seasonal_capacity)
    model.seasonal_soc_limit = pyo.Constraint(model.T, rule=seasonal_soc_limit)

    SUMMER = set(range(2880, 6552))  # April–September (Stunden 2880–6552)

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
        raise RuntimeError(f"Solver nicht erfolgreich: {results.solver.termination_condition}")

    # Ergebnisse

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

    print(f'Die Speichergröße beträgt {storage_cap_res:.1f} m3 bzw. {storage_volume_to_MWh(storage_cap_res):.1f} MWh')
    print(f'Die Wärmepumpengröße beträgt {hp_capacity_res:.1f} MW')
    print(f'Die Gaskessel-Kapazität beträgt {gas_boiler_cap_res:.1f} MW')
    print(f'Gasanteil Jahreswärme: {Q_gas_boiler_res.sum() / sum(demand) * 100:.1f} %')
    print(f'PV-Kapazität beträgt {pv_cap_res:.1f} MW')
    print(f'PV-Einspeisung beträgt {pv_feed_in_res.sum():.1f} MWh mit einem Ertrag von {pv_feed_in_res.sum() * feed_in_tariff:.1f} Euro')
    print(f'PV-Verfügbarkeit: {pv_res.sum():.1f} MWh')
    seasonal_cap_MWh = storage_volume_to_MWh(seasonal_cap_res)
    print(f'Saisonalspeicher Kapazität:    {seasonal_cap_res:.1f} m³ = {seasonal_cap_MWh:.2f} MWh')
    print(f'Saisonalspeicher Peak-SOC:     {seasonal_soc_res.max():.2f} MWh  ({seasonal_soc_res.max()/seasonal_cap_MWh*100:.0f}% der Kapazität)' if seasonal_cap_MWh > 0 else '')
    print(f'Saisonalspeicher Einspeisung:  {seasonal_charge_res.sum():.2f} MWh/a')
    print(f'Saisonalspeicher Ausspeisung:  {seasonal_discharge_res.sum():.2f} MWh/a')

    return (results, Q_hp_res, Q_gas_boiler_res, charge_res, discharge_res, SOC_res,
            storage_cap_res, gas_boiler_cap_res,
            pv_res, pv_feed_in_res, pv_cap_res,
            seasonal_charge_res, seasonal_discharge_res, seasonal_soc_res, seasonal_cap_res)

