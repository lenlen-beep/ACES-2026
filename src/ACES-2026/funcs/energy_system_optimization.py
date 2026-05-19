import pyomo.environ as pyo
import numpy as np

from funcs.read_data import read_parameters

    # TODO:Der Speicher kann bei SOC = 0 keine Wärme verlieren (theoretisch aber schon, da 55°C RLT) 
    # --> beachten :)


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
cp_W = parameters["Net_parameters"]["specific_heat_capacity"]
rho_W = parameters["Net_parameters"]["density"]
s_temp = parameters["Net_parameters"]["supply_temperature"]
delta_T = parameters["Net_parameters"]["delta_T"]

#Investitionsparameter
r = parameters["Invest_parameters"]["interest_rate"]
n = parameters["Invest_parameters"]["lifetime_years"]
storage_invest_offset = parameters["System_parameters"]["Storage"]["invest_offset"]
storage_specific_cost = parameters["System_parameters"]["Storage"]["specific_cost"]

annuity_factor = r * (1+r)**n / ((1+r)**n - 1)

#Pumpenparameter
g = parameters["System_parameters"]["Pump"]["gravity"]
h = parameters["System_parameters"]["Pump"]["del_height"]
eta_p = parameters["System_parameters"]["Pump"]["eta_pump"]

P_pump = (g * h) / (eta_p * cp_W*1000 *delta_T) #MW

#Wärmepumpenparameter
Pth_wp_max = parameters["System_parameters"]["HP"]["hp_thermal_power_max"]
COP = parameters["System_parameters"]["HP"]["COP"]

#Speicherparameter
max_charge_rate = parameters["System_parameters"]["Storage"]["max_charge_rate"]
max_discharge_rate = parameters["System_parameters"]["Storage"]["max_discharge_rate"]
p_loss = parameters["System_parameters"]["Storage"]["p_loss"]
SOC_init = parameters["System_parameters"]["Storage"]["SOC_init"]
storage_cap = parameters["System_parameters"]["Storage"]["initial_storage_capacity"]

def storage_volume_to_MWh(vol_m3):
        # vol_m3 = m3 
        # mass_kg = vol_m3 * rho_W
        # energy_kJ = mass_kg * cp_W * delta_T
        # energy_kWh = energy_kJ / 3600
        # energy_MWh = energy_kWh / 1000
    return (vol_m3) * rho_W * cp_W * delta_T / (3600.0) #MWh


def optimize_energy_system(demand, price):
    
    demand = demand.values

    T = range(len(demand))

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
    model.P_wp = pyo.Var(model.T, bounds=(0, Pth_wp_max)) #MW

    # Speicher
    model.charge = pyo.Var(model.T, bounds=(0, None)) #MW
    model.discharge = pyo.Var(model.T, bounds=(0, None)) #MW
    model.SOC = pyo.Var(model.T, bounds=(0, None)) #kWh
    model.storage_capacity = pyo.Var(bounds=(0, None), initialize=storage_cap) #m3


    # --------------------------------
    # Zielfunktion
    # --------------------------------

    # Stromkosten minimieren
    def obj_rule(m):
        # Preis Strom [t] * Leistung WP [t] + Investkosten Speicher 
        # (linearisiert: offset + spezifische Kosten * Kapazität) 
        # --> umgerechnet auf jährliche Kosten mit Annuitätenfaktor
        return sum(price[t] * (m.P_wp[t]/COP) for t in m.T) + (storage_invest_offset + \
                storage_specific_cost * (m.storage_capacity)) * annuity_factor + P_pump * \
                sum(price[t] * (m.charge[t] + m.discharge[t]) for t in m.T)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


    # --------------------------------
    # Constraints
    # --------------------------------

    # Wärmebilanz
    def heat_balance(m, t):
        #Verfügbare Wärmeleistung (P_WP + P_SP) == Benötigte Wärmeleistung (Last + P_SP)
        return m.P_wp[t] + m.discharge[t] == demand[t] + m.charge[t]

    model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)


    # Speicher-Dynamik
    def storage_rule(m, t):
        # Anfangs- und End-SOC auf x% initialen Ladezustand setzen, um realistische 
        # Betriebsbedingungen zu gewährleisten (z.B. Vermeidung von unrealistischen 
        # Szenarien mit dauerhaft leerem oder vollem Speicher).
        #Restliche Zeitschritte: SOC immer gleich dem SOC aus vorherigen Zeitschritt 
        # + charge oder - discharge - Verluste
        storage_MWh = storage_volume_to_MWh(storage_cap)
        p_loss_MWh = p_loss
        
        if t == 0:
            return m.SOC[t] == SOC_init * storage_MWh #x% initialer Ladezustand (Anfang)
        
        elif t == T[-1]:
            return m.SOC[t] == SOC_init * storage_MWh #x% initialer Ladezustand (Ende)
        
        return m.SOC[t] == m.SOC[t-1] + m.charge[t] - m.discharge[t] -p_loss_MWh

    model.storage = pyo.Constraint(model.T, rule=storage_rule)


    def soc_capacity_limit(m, t):
        # Speicher kann nicht mehr Energie speichern als die Kapazität erlaubt
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        return m.SOC[t] <= storage_MWh

    model.soc_capacity_limit = pyo.Constraint(model.T, rule=soc_capacity_limit)


    def charge_power_limit_storage(m, t):
        # Die Ladungsleistung ist begrenzt durch die maximale Ladungsrate 
        # und die Speicherkapazität
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)

        return m.charge[t] <= max_charge_rate * storage_MWh
    
    model.charge_power_limit_storage = pyo.Constraint(model.T, rule=charge_power_limit_storage)


    def charge_power_limit_hp(m, t):
        # Die Ladungsleistung ist begrenzt durch die maximale Leistung der Wärmepumpe, 
        # da die Ladung nur über die WP erfolgen kann
        return m.charge[t] <= Pth_wp_max

    model.charge_power_limit_hp = pyo.Constraint(model.T, rule=charge_power_limit_hp)


    def discharge_power_limit(m, t):
        # Die Entladungsleistung ist begrenzt durch die maximale Entladungsrate
        # und die Speicherkapazität
        storage_MWh = storage_volume_to_MWh(m.storage_capacity)
        return m.discharge[t] <= max_discharge_rate * storage_MWh

    model.discharge_power_limit = pyo.Constraint(model.T, rule=discharge_power_limit)
    

    def negative_price_discharge_restrict(m, t):
        # Wenn der Strompreis negativ ist, soll die Entladung des Speichers auf 0 
        # begrenzt werden, um zu verhindern, dass der Speicher in Zeiten negativer 
        # Preise entlädt (was wirtschaftlich nicht sinnvoll wäre und zu Zyklen führt).
        if price[t] < 0:
            return m.discharge[t] == 0 
        else:
            return pyo.Constraint.Skip
        
    model.negative_price_discharge_restrict = pyo.Constraint(model.T, rule=negative_price_discharge_restrict)


    # Solver
    solver = pyo.SolverFactory('glpk')
    results = solver.solve(model)

    # Ergebnisse 
    P_wp_res = np.array([pyo.value(model.P_wp[t]) for t in T])
    charge_res = np.array([pyo.value(model.charge[t]) for t in T])
    discharge_res = np.array([pyo.value(model.discharge[t]) for t in T])
    SOC_res = np.array([pyo.value(model.SOC[t]) for t in T])
    storage_cap_res = pyo.value(model.storage_capacity)

    print(f'Die Speichergröße beträgt {storage_cap_res} m3 bzw. {storage_cap_res * cp_W * delta_T /(3600)} kWh')

    return results, P_wp_res, charge_res, discharge_res, SOC_res, storage_cap_res

