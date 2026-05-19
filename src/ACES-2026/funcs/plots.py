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
# BDEW Lastprofile plotten
# --------------------------------------------------

def plot_bdew_profiles(profiles, total_load, show_plot=True):
    plt.figure(figsize=(12,5))

    for name, profile in profiles.items():
        plt.plot(profile, label=name, alpha=0.7)

    plt.plot(
        total_load,
        label="Total",
        linewidth=2
    )

    plt.legend()
    plt.grid()
    plt.title("BDEW Heat Profiles")

    plt.xlabel("Zeit in h")
    plt.ylabel("Leistung in MW")

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
# Plots des Energiesystems
# --------------------------------------------------

def plot_energy_system_output_sorted(demand, P_wp_res, discharge_res, show_plot=True):
    demand = np.array(demand)
    sorted_idx = np.argsort(-demand)

    demand_sorted = demand[sorted_idx]
    wp_sorted = (P_wp_res)[sorted_idx]
    discharge_sorted = discharge_res[sorted_idx]

    #Sortiert Last WP u. Speicher
    plt.figure()
    plt.plot(demand_sorted, label="Wärmebedarf")
    plt.plot(wp_sorted, label="Wärmepumpe")
    plt.plot(discharge_sorted, label="Speicher Entladung")
    plt.xlabel("Stunden (sortiert)")
    plt.ylabel("Leistung [MW]")
    plt.title("Dauerlinie mit Einsatz von WP und Speicher")
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


def plot_load_w_components(P_wp_res, discharge_res, demand, show_plot=True):

    # Dauerlinie, Last WP + Speicher
    plt.figure()
    plt.plot(P_wp_res, label="Wärmepumpe")
    plt.plot(discharge_res, label="Speicher Entladung")
    plt.plot(demand, label="Wärmebedarf")
    plt.xlabel("Zeit [h]")
    plt.ylabel("Leistung [MW]")
    plt.title("Zeitreihe: Einsatz von WP und Speicher")
    plt.legend()
    plt.grid()
    
    if show_plot:
        plt.show()


def plot_SOC(SOC_res, storage_cap_res, show_plot=True):
    # --- Plot SOC ---
    plt.figure()
    plt.plot(SOC_res, label="SOC (model units)")
    # if storage_capacity is set, show capacity line and fraction
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

