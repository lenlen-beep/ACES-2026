import os
import matplotlib.pyplot as plt
import numpy as np

from funcs.energy_system_optimization import storage_volume_to_MWh

# Calibri-ähnliche Schrift (Calibri → Helvetica → Arial als Fallback)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

# Farbpalette Energiesystem
COLOR_WP       = "#00395B"   # EUF-Blau      — Wärmepumpe
COLOR_GAS      = "#C17A2F"   # Amber         — Gaskessel (fossil)
COLOR_SPEICHER = "#769D7B"   # Mint          — Kurzzeitspeicher
COLOR_SAISONAL = "#2F6B4F"   # Dunkelgrün    — Saisonalspeicher
COLOR_LAST     = "#1A1A1A"   # Fast-Schwarz  — Wärmebedarf (Referenz)
COLOR_VERLUST  = "#A0463A"   # Gedämpftes Rot — Netzverluste
COLOR_PV       = "#C8A84B"   # Gold          — PV / Sonne

PLOTS_DIR = "src/ACES-2026/plots"

LABEL_FONTSIZE = 15
TICK_FONTSIZE  = 13
LEGEND_FONTSIZE = 13
TITLE_FONTSIZE = 16

def _ppt_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.2, color="#CCCCCC")
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.margins(x=0)

def _save(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {path}")


# --------------------------------------------------
# Temperaturen plotten
# --------------------------------------------------
def plot_temperatures(temperature, station_id, show_plot=True):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.plot(temperature, color=COLOR_WP, linewidth=0.8)
    ax.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Temperature in °C", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Hourly outdoor temperature 2019 – Station {station_id}",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")
    _ppt_style(ax)
    fig.tight_layout()
    if show_plot:
        plt.show()


# --------------------------------------------------
# Strompreise plotten
# --------------------------------------------------
def plot_prices(prices, show_plot=True):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.plot(prices, color=COLOR_WP, linewidth=0.8)
    ax.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Electricity price in €/MWh", fontsize=LABEL_FONTSIZE)
    ax.set_title("Hourly day-ahead electricity prices 2024",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")
    _ppt_style(ax)
    fig.tight_layout()
    if show_plot:
        plt.show()


# --------------------------------------------------
# Gaspreise plotten
# --------------------------------------------------
def plot_gas_prices(gas_prices, show_plot=True):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.plot(gas_prices, color=COLOR_GAS, linewidth=0.8)
    ax.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Gas price in €/MWh", fontsize=LABEL_FONTSIZE)
    ax.set_title("Daily PEGAS THE DA settlement prices 2024 (hourly interpolated)",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")
    _ppt_style(ax)
    fig.tight_layout()
    if show_plot:
        plt.show()


# --------------------------------------------------
# Plots des Energiesystems
# --------------------------------------------------

def plot_energy_system_output_sorted(demand, P_wp_res, discharge_res, gas_boiler_res,
                                     seasonal_discharge_res=None, show_plot=True):
    demand     = np.array(demand)
    sorted_idx = np.argsort(-demand)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.plot(demand[sorted_idx],                                color=COLOR_LAST,     linewidth=1.8, label="Heat demand")
    ax.plot(np.array(P_wp_res)[sorted_idx],                   color=COLOR_WP,       linewidth=1.2, label="Heat pump")
    ax.plot(np.array(discharge_res)[sorted_idx],              color=COLOR_SPEICHER, linewidth=1.2, label="Buffer storage discharge")
    ax.plot(np.array(gas_boiler_res)[sorted_idx],             color=COLOR_GAS,      linewidth=1.2, label="Gas boiler")
    if seasonal_discharge_res is not None:
        ax.plot(np.array(seasonal_discharge_res)[sorted_idx], color=COLOR_SAISONAL, linewidth=1.2,
                label="Seasonal storage discharge")
    ax.set_xlabel("Hours (sorted by heat demand)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Power in MW", fontsize=LABEL_FONTSIZE)
    ax.set_title("Ordered annual load duration curve – Heat supply",
                 fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax)
    fig.tight_layout()
    _save(fig, "dauerlinie_energiesystem.png")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_charge_discharge_process(charge_res, discharge_res, SOC_res, storage_cap_res, show_plot=True):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 9), facecolor="white")
    fig.subplots_adjust(hspace=0.08)

    ax1.plot(charge_res,    color=COLOR_SPEICHER, linewidth=1.0, label="Charging")
    ax1.plot(discharge_res, color=COLOR_WP,       linewidth=1.0, label="Discharging")
    ax1.set_ylabel("Power in MW", fontsize=LABEL_FONTSIZE)
    ax1.set_title("Buffer storage, Charging and discharging",
                  fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax1.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax1)

    ax2.plot(SOC_res, color=COLOR_SAISONAL, linewidth=1.0, label="SOC")
    if storage_cap_res is not None and storage_cap_res > 0:
        storage_MWh = storage_volume_to_MWh(storage_cap_res)
        ax2.axhline(storage_MWh, color="#888888", linestyle="--", linewidth=1.2,
                    label=f"Capacity {storage_MWh:.1f} MWh")
    ax2.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Energy in MWh", fontsize=LABEL_FONTSIZE)
    ax2.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax2)

    fig.tight_layout()
    _save(fig, "pufferspeicher_laden_entladen.png")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_load_w_components(P_wp_res, discharge_res, gas_boiler_res, demand,
                           seasonal_discharge_res=None, show_plot=True):
    demand = np.array(demand)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.plot(P_wp_res,       color=COLOR_WP,       linewidth=1.0, label="Heat pump")
    ax.plot(discharge_res,  color=COLOR_SPEICHER,  linewidth=1.0, label="Buffer storage discharge")
    if seasonal_discharge_res is not None:
        ax.plot(seasonal_discharge_res, color=COLOR_SAISONAL, linewidth=1.0,
                label="Seasonal storage discharge")
    ax.plot(gas_boiler_res, color=COLOR_GAS,      linewidth=1.0, label="Gas boiler")
    ax.plot(demand,         color=COLOR_LAST,     linewidth=1.8, label="Heat demand")
    ax.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Power in MW", fontsize=LABEL_FONTSIZE)
    ax.set_title("Component distribution on the heat supply", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax)
    fig.tight_layout()
    _save(fig, "zeitreihe_energiesystem.png")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_SOC(SOC_res, storage_cap_res, show_plot=True):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.plot(SOC_res, color=COLOR_SPEICHER, linewidth=1.0, label="Buffer storage SOC")
    if storage_cap_res is not None and storage_cap_res > 0:
        storage_MWh = storage_volume_to_MWh(storage_cap_res)
        ax.axhline(storage_MWh, color="#888888", linestyle="--", linewidth=1.2,
                   label=f"Capacity {storage_MWh:.1f} MWh")
    ax.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Energy in MWh", fontsize=LABEL_FONTSIZE)
    #ax.set_title("State of Charge – Buffer storage", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc='upper right')
    ax.set_ylim(top=3)
    _ppt_style(ax)
    fig.tight_layout()
    _save(fig, "soc_pufferspeicher.png")
    if show_plot:
        plt.show()
    plt.close(fig)


# --------------------------------------------------
# Netzverluste plotten
# --------------------------------------------------

def plot_network_losses(result_df, show_plot=True,
                        out_path="src/ACES-2026/plots/netzverluste.png"):
    C_GESAMT   = "#2F6B4F"   # Dunkelgrün — network feed-in
    C_GEBAEUDE = "#769D7B"   # Mint       — building consumption
    C_VERLUST  = "#A0463A"   # Rot        — network losses

    total_kw     = result_df['load_kW'].values
    consumer_kw  = result_df['consumer_load_kW'].values
    loss_kw      = result_df['net_loss_kW'].values
    hours        = np.arange(len(total_kw))

    jahres_MWh   = total_kw.sum()    / 1000
    gebaeude_MWh = consumer_kw.sum() / 1000
    verlust_MWh  = loss_kw.sum()     / 1000
    anteil_pct   = verlust_MWh / jahres_MWh * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, facecolor="white")
    fig.subplots_adjust(hspace=0.08, top=0.88, bottom=0.10, left=0.09, right=0.97)

    #fig.suptitle("Heat losses in the district heating network",
                 #fontsize=20, fontweight="bold", color="#1A1A1A")
    #fig.text(0.5, 0.91,
             #f"Annual total: {jahres_MWh:,.0f} MWh/a  ·  "
             #f"Network losses: {verlust_MWh:,.0f} MWh/a  ·  Share: {anteil_pct:.1f} %",
             #ha="center", fontsize=LEGEND_FONTSIZE, color="#555555")

    ax1.plot(hours, total_kw / 1000, color=C_GESAMT,   linewidth=1.5, label="Network feed-in")
    ax1.plot(hours, consumer_kw / 1000, color=C_GEBAEUDE, linewidth=1.5, label="Building consumption")
    ax1.plot(hours, loss_kw / 1000, color=C_VERLUST, linewidth=1.2, label="Network losses")
    ax1.set_ylabel("Power in MW", fontsize=LABEL_FONTSIZE)
    ax1.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper center", facecolor="White")
    ax1.grid(True, alpha=0.2, color="#CCCCCC")
    ax1.set_title("Power distribution in the district heating network", fontsize=TITLE_FONTSIZE - 1)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.tick_params(labelsize=TICK_FONTSIZE, labelbottom=False)
    ax1.margins(x=0)

    ax2.plot(hours, loss_kw / 1000, color=C_VERLUST, linewidth=1.0)
    #ax2.axhline(loss_kw.mean() / 1000, color="#1A1A1A", linestyle="--",
                #linewidth=1.2, label=f"Mean  {loss_kw.mean()/1000:.3f} MW")
    ax2.set_ylabel("Network losses in MW", fontsize=LABEL_FONTSIZE)
    ax2.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax2.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    ax2.grid(True, alpha=0.2, color="#CCCCCC")
    #ax2.set_title("Net losses", fontsize=TITLE_FONTSIZE - 1)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.tick_params(labelsize=TICK_FONTSIZE)
    ax2.margins(x=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


# --------------------------------------------------
# PV plotten
# --------------------------------------------------

def plot_pv(pv_res, pv_feed_in_res, pv_cap_res, show_plot=True):
    pv_self_use = np.array(pv_res) - np.array(pv_feed_in_res)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(16, 9),
                                   facecolor="white")
    fig.subplots_adjust(hspace=0.12)

    ax1.plot(pv_res,         color=COLOR_PV,  linewidth=0.8, label="PV generation")
    ax1.plot(pv_self_use,    color=COLOR_WP,  linewidth=0.8, label="Self-consumption (heat pump)")
    ax1.plot(pv_feed_in_res, color=COLOR_GAS, linewidth=0.8, label="Grid feed-in")
    ax1.set_ylabel("Power in MW", fontsize=LABEL_FONTSIZE)
    ax1.set_title(f"PV system – Time series (capacity: {pv_cap_res:.1f} MW)",
                  fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax1.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax1)

    sorted_pv = np.sort(np.array(pv_res))[::-1]
    ax2.fill_between(range(len(sorted_pv)), sorted_pv, color=COLOR_PV, alpha=0.5,
                     label="PV generation (sorted)")
    ax2.set_xlabel("Hours (sorted)", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Power in MW", fontsize=LABEL_FONTSIZE)
    ax2.set_title("PV load duration curve", fontsize=TITLE_FONTSIZE - 1)
    ax2.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax2)

    fig.tight_layout()
    _save(fig, "pv_erzeugung.png")
    if show_plot:
        plt.show()
    plt.close(fig)


# --------------------------------------------------
# Saisonaler Speicher plotten
# --------------------------------------------------

def plot_seasonal_storage(seasonal_charge_res, seasonal_discharge_res, seasonal_soc_res,
                          seasonal_cap_res, show_plot=True):
    seasonal_MWh = storage_volume_to_MWh(seasonal_cap_res) if seasonal_cap_res > 0 else 0

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 9),
                                   facecolor="white")
    fig.subplots_adjust(hspace=0.08)

    ax1.plot(seasonal_charge_res,    color=COLOR_SAISONAL, linewidth=0.8, label="Charging")
    ax1.plot(seasonal_discharge_res, color=COLOR_GAS,      linewidth=0.8, label="Discharging")
    ax1.set_ylabel("Power in MW", fontsize=LABEL_FONTSIZE)
    ax1.set_title("Seasonal storage, Charging and discharging",
                  fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax1.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax1)

    ax2.plot(seasonal_soc_res, color=COLOR_SAISONAL, linewidth=0.8, label="SOC")
    if seasonal_MWh > 0:
        ax2.axhline(seasonal_MWh, color="#888888", linestyle="--", linewidth=1.2,
                    label=f"Capacity {seasonal_MWh:.0f} MWh / {seasonal_cap_res:.0f} m³")
    ax2.set_xlabel("Time in h", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Energy in MWh", fontsize=LABEL_FONTSIZE)
    ax2.legend(fontsize=LEGEND_FONTSIZE, frameon=False)
    _ppt_style(ax2)

    fig.tight_layout()
    _save(fig, "saisonalspeicher.png")
    if show_plot:
        plt.show()
    plt.close(fig)
