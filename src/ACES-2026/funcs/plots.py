import matplotlib.pyplot as plt
import numpy as np

from funcs.energy_system_optimization import storage_volume_to_MWh

# --------------------------------------------------
# Temperaturen plotten
# --------------------------------------------------
def plot_temperatures(temperature, station_id, show_plot=True):

    plt.figure(figsize=(12, 4))
    plt.plot(temperature, color="steelblue", linewidth=0.8)
    plt.xlabel("Zeit in h")
    plt.ylabel("Temperatur in °C")
    plt.title(f"Stündliche Außentemperatur 2024 an der Station {station_id}")
    plt.grid(True)
    plt.tight_layout()

    if show_plot:
        plt.show()



# --------------------------------------------------
# Strompreise plotten
# --------------------------------------------------
def plot_prices(prices, show_plot=True):

    plt.figure(figsize=(12, 4))
    plt.plot(prices, color="steelblue", linewidth=0.8)
    plt.xlabel("Zeit in h")
    plt.ylabel("Strompreis in €/MWh")
    plt.title(f"Stündliche Großhandelsstrompreise 2024")
    plt.grid(True)
    plt.tight_layout()

    if show_plot:
        plt.show()


# --------------------------------------------------
# Gaspreise plotten
# --------------------------------------------------
def plot_gas_prices(gas_prices, show_plot=True):

    plt.figure(figsize=(12, 4))
    plt.plot(gas_prices, color="tomato", linewidth=0.8)
    plt.xlabel("Zeit in h")
    plt.ylabel("Gaspreis in €/MWh")
    plt.title("Tägliche PEGAS THE DA Settlement-Preise 2024 (stündlich interpoliert)")
    plt.grid(True)
    plt.tight_layout()

    if show_plot:
        plt.show()



# --------------------------------------------------
# Plots des Energiesystems
# --------------------------------------------------

def plot_energy_system_output_sorted(demand, P_wp_res, discharge_res, gas_boiler_res,
                                     seasonal_discharge_res=None, show_plot=True):
    demand = np.array(demand)
    sorted_idx = np.argsort(-demand)

    plt.figure()
    plt.plot(demand[sorted_idx], label="Wärmebedarf")
    plt.plot(P_wp_res[sorted_idx], label="Wärmepumpe")
    plt.plot(discharge_res[sorted_idx], label="Kurzzeitspeicher Entladung")
    plt.plot(gas_boiler_res[sorted_idx], label="Gaskessel", color="tomato")
    if seasonal_discharge_res is not None:
        plt.plot(seasonal_discharge_res[sorted_idx], label="Saisonalspeicher Entladung", color="darkorange")
    plt.xlabel("Stunden (sortiert)")
    plt.ylabel("Leistung [MW]")
    plt.title("Dauerlinie mit Einsatz von WP, Speicher und Gaskessel")
    plt.legend()
    plt.grid()

    if show_plot:
        plt.show()


def plot_charge_discharge_process(charge_res, discharge_res, SOC_res, storage_cap_res, show_plot=True):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

    # oben: Laden / Entladen
    ax1.plot(charge_res, label="Laden")
    ax1.plot(discharge_res, label="Entladen")
    ax1.set_ylabel("Leistung [MW]")
    ax1.set_title("Speicher Lade- und Entladevorgänge")
    ax1.legend()
    ax1.grid(True)

    # unten: SOC
    ax2.plot(SOC_res, color="tab:green", label="SOC")
    if storage_cap_res is not None and storage_cap_res > 0:
        storage_MWh = storage_volume_to_MWh(storage_cap_res)
        ax2.plot(np.full(len(SOC_res), storage_MWh), '--', color="gray", label="Kapazität (MWh)")
    ax2.set_xlabel("Zeit [h]")
    ax2.set_ylabel("Energie [MWh]")
    ax2.set_title("State of Charge (SOC)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    if show_plot:
        plt.show()


def plot_load_w_components(P_wp_res, discharge_res, gas_boiler_res, demand,
                           seasonal_discharge_res=None, show_plot=True):
    demand = np.array(demand)
    plt.figure()
    plt.plot(P_wp_res, label="Wärmepumpe")
    plt.plot(discharge_res, label="Kurzzeitspeicher Entladung")
    if seasonal_discharge_res is not None:
        plt.plot(seasonal_discharge_res, label="Saisonalspeicher Entladung", color="darkorange")
    plt.plot(gas_boiler_res, label="Gaskessel", color="tomato")
    plt.plot(demand, label="Wärmebedarf", color="black", linewidth=1.2)
    plt.xlabel("Zeit [h]")
    plt.ylabel("Leistung [MW]")
    plt.title("Zeitreihe: Einsatz von WP, Speicher und Gaskessel")
    plt.legend()
    plt.grid()

    if show_plot:
        plt.show()


def plot_SOC(SOC_res, storage_cap_res, show_plot=True):
    plt.figure()
    plt.plot(SOC_res, label="SOC (model units)")
    if storage_cap_res is not None and storage_cap_res > 0:
        storage_MWh = storage_volume_to_MWh(storage_cap_res)
        plt.plot(np.full(len(SOC_res), storage_MWh), '--', label="Capacity (MWh)")
        plt.figure()
        plt.plot(SOC_res / storage_MWh, label="SOC / Capacity")
        plt.xlabel("Time [h]")
        plt.ylabel("Fraction of capacity")
        plt.title("SOC as fraction of capacity")
        plt.legend()
        plt.grid()

    plt.xlabel("Time [h]")
    plt.ylabel("SOC (kWh) / model units")
    plt.title("State of Charge (SOC) over time")
    plt.legend()
    plt.grid()

    if show_plot:
        plt.show()


# --------------------------------------------------
# PV plotten
# --------------------------------------------------

def plot_pv(pv_res, pv_feed_in_res, pv_cap_res, show_plot=True):
    pv_self_use = pv_res - pv_feed_in_res

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(12, 6))

    ax1.plot(pv_res, label="PV Erzeugung", color="gold", linewidth=0.8)
    ax1.plot(pv_self_use, label="Eigenverbrauch (Wärmepumpe)", color="steelblue", linewidth=0.8)
    ax1.plot(pv_feed_in_res, label="Netzeinspeisung", color="tomato", linewidth=0.8)
    ax1.set_ylabel("Leistung [MW]")
    ax1.set_title(f"PV-Anlage – Zeitreihe (Kapazität: {pv_cap_res:.1f} MW)")
    ax1.legend()
    ax1.grid(True)

    sorted_pv = np.sort(pv_res)[::-1]
    ax2.fill_between(range(len(sorted_pv)), sorted_pv, alpha=0.6, color="gold", label="PV Erzeugung (sortiert)")
    ax2.set_xlabel("Stunden (sortiert)")
    ax2.set_ylabel("Leistung [MW]")
    ax2.set_title("PV-Dauerlinie")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if show_plot:
        plt.show()


# --------------------------------------------------
# Saisonaler Speicher plotten
# --------------------------------------------------

def plot_seasonal_storage(seasonal_charge_res, seasonal_discharge_res, seasonal_soc_res,
                          seasonal_cap_res, show_plot=True):
    seasonal_MWh = storage_volume_to_MWh(seasonal_cap_res) if seasonal_cap_res > 0 else 0

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 6))

    ax1.plot(seasonal_charge_res, label="Laden (Sommer)", color="steelblue", linewidth=0.8)
    ax1.plot(seasonal_discharge_res, label="Entladen (Winter)", color="tomato", linewidth=0.8)
    ax1.set_ylabel("Leistung [MW]")
    ax1.set_title("Saisonaler Speicher – Lade- und Entladevorgänge")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(seasonal_soc_res, label="SOC", color="tab:green", linewidth=0.8)
    if seasonal_MWh > 0:
        ax2.plot(np.full(len(seasonal_soc_res), seasonal_MWh), '--', color="gray",
                 label=f"Kapazität ({seasonal_MWh:.0f} MWh / {seasonal_cap_res:.0f} m³)")
    ax2.set_xlabel("Zeit [h]")
    ax2.set_ylabel("Energie [MWh]")
    ax2.set_title("Saisonaler SOC")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if show_plot:
        plt.show()

