"""
Berechnung der Wärmegestehungskosten (LCOH – Levelized Cost of Heat).

Alle Kostenpositionen des Fernwärmesystems werden einzeln berechnet – als
annualisierte Investitionen (CAPEX) bzw. als jährliche Betriebskosten (OPEX) –
und anschließend zu einem Gesamt-LCOH zusammengeführt:

    LCOH = ( Σ annualisierte CAPEX + Σ jährliche OPEX − PV-Erlös )
           ----------------------------------------------------------   [€/MWh]
                        gelieferte Jahreswärme

Komponenten:
    CAPEX : Wärmepumpe, Pufferspeicher, Gaskessel, PV, Saisonalspeicher, Wärmenetz
    OPEX  : Strombezug (WP), Pumpstrom, Gas-Brennstoff
    Erlös : PV-Einspeisung (Gutschrift, senkt den LCOH)

Zusätzlich wird ein Kreisdiagramm erstellt, das den prozentualen Anteil jeder
Technologie / Kostenposition am LCOH zeigt.

Verwendung (z. B. in main.py nach der Optimierung):

    from LCOH import calculate_lcoh, plot_lcoh_pie

    lcoh, components = calculate_lcoh(
        demand=load,
        electricity_price=electricity_price,
        gas_price=gas_price,
        Q_hp=result_df_heatpump,
        charge=result_df_charge,
        discharge=result_df_discharge,
        Q_gas_boiler=result_df_gas_boiler,
        pv_availability=result_pv,
        pv_feed_in=result_pv_feed_in,
        storage_capacity_m3=result_storage_capacity,
        gas_boiler_capacity=result_gas_boiler_capacity,
        pv_capacity=result_pv_capacity,
        seasonal_capacity_m3=result_seasonal_capacity,
        network_length=network_length,          # [m] aus gdf["Length"].sum()
        elec_price_mode="spot",                  # identisch zur Optimierung wählen!
    )
    plot_lcoh_pie(components, lcoh, show_plot=True)
"""

import os

import numpy as np
import matplotlib.pyplot as plt

# Arbeitsverzeichnis auf das Repo-Root setzen, damit die im Projekt hartcodierten
# relativen Pfade ("src/ACES-2026/parameters.yaml") in funcs.* und hier auch dann
# funktionieren, wenn das Skript direkt (z. B. via Spyder %runfile --wdir) aus dem
# Skript-Ordner gestartet wird. LCOH.py liegt in <repo>/src/ACES-2026/funcs/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(_REPO_ROOT)

from funcs.read_data import read_parameters
from funcs.energy_system_optimization import (
    annuity_factor,
    P_pump,
    COP,
    eta_gas_boiler,
    feed_in_tariff,
    hp_invest_offset, hp_specific_cost,
    storage_invest_offset, storage_specific_cost,
    gas_invest_offset, gas_specific_cost,
    pv_invest_offset, pv_specific_cost,
    seasonal_invest_offset, seasonal_specific_cost,
)

parameters = read_parameters("src/ACES-2026/parameters.yaml")

# Netz-/Rohrparameter (CAPEX über Netzlänge) – einziger Posten, der in der
# Optimierung bisher nicht enthalten ist.
pipe_specific_cost = parameters["pipe_parameters"]["specific_invest_pipe"]  # €/m


def _to_array(x):
    """pandas.Series oder array-artig → 1D-float-numpy-Array."""
    return np.asarray(getattr(x, "values", x), dtype=float)


def calculate_lcoh(
    demand,
    electricity_price,
    gas_price,
    Q_hp,
    charge,
    discharge,
    Q_gas_boiler,
    pv_availability,
    pv_feed_in,
    storage_capacity_m3,
    gas_boiler_capacity,
    pv_capacity,
    seasonal_capacity_m3,
    network_length,
    hp_capacity=None,
    P_grid=None,
    cop=None,
    elec_price_mode: str = "spot",
    elec_hedge_share: float = 0.0,
    gas_price_mode: str = "spot",
    verbose: bool = True,
):
    """
    Berechnet den LCOH und alle Einzelkomponenten.

    Die Preis-Modi (elec_price_mode / gas_price_mode / elec_hedge_share) MÜSSEN
    identisch zu denen der Optimierung gewählt werden, sonst passen Betriebs-
    kosten und Dispatch nicht zusammen.

    hp_capacity : optional. Wird bei None aus dem Dispatch rekonstruiert
                  (= max(Q_hp, charge); im Kostenoptimum exakt, da die HP-Kapazität
                  minimiert und nur durch Q_hp <= cap und charge <= cap begrenzt ist).
    P_grid      : optional. Wird bei None aus der Strombilanz rekonstruiert
                  (P_grid = Q_hp/COP + pv_feed_in − pv_availability).
    cop         : optional, array-like (gleiche Länge wie demand) – MUSS derselbe
                  COP-Verlauf sein, mit dem optimiert wurde (sonst passt P_el_hp
                  nicht zum Dispatch). Bei None wird der statische COP aus
                  parameters.yaml verwendet (bisheriges Verhalten).

    Rückgabe:
        lcoh_total : float, €/MWh
        components : dict[label] = {"eur_per_year", "eur_per_mwh", "share_pct"}
                     (PV-Erlös ist negativ; share_pct bezieht sich auf den
                      Netto-LCOH-Gesamtbetrag.)
    """
    demand            = _to_array(demand)
    electricity_price = _to_array(electricity_price)
    gas_price         = _to_array(gas_price)
    Q_hp              = _to_array(Q_hp)
    charge            = _to_array(charge)
    discharge         = _to_array(discharge)
    Q_gas_boiler      = _to_array(Q_gas_boiler)
    pv_availability   = _to_array(pv_availability)
    pv_feed_in        = _to_array(pv_feed_in)

    n_t = len(demand)

    # --- Preisreihen aufbereiten (identisch zur Optimierung) -----------------
    elec_tariff = parameters["price_parameters"]["electricity"]["tarif"]["usual_mid"] * 10  # ct/kWh → €/MWh
    if elec_price_mode == "tariff":
        electricity_price = np.full(n_t, elec_tariff)
    elif elec_price_mode == "hedge":
        electricity_price = elec_hedge_share * elec_tariff + (1 - elec_hedge_share) * electricity_price
    # "spot" → unverändert

    gas_tariff = parameters["price_parameters"]["gas"]["tarif"]["usual_mid"] * 10  # ct/kWh → €/MWh
    if gas_price_mode == "tariff":
        gas_price = np.full(n_t, gas_tariff)
    # "spot" → unverändert

    # --- fehlende Größen rekonstruieren --------------------------------------
    if hp_capacity is None:
        hp_capacity = max(float(np.max(Q_hp)) if Q_hp.size else 0.0,
                          float(np.max(charge)) if charge.size else 0.0)

    cop_arr = np.full(n_t, COP) if cop is None else _to_array(cop)
    P_el_hp = Q_hp / cop_arr
    if P_grid is None:
        # Strombilanz: P_grid + pv_avail = P_el_hp + pv_feed_in
        P_grid = P_el_hp + pv_feed_in - pv_availability
        P_grid = np.clip(P_grid, 0.0, None)  # Rundungsreste abfangen

    # --- CAPEX (annualisiert, €/a) -------------------------------------------
    # Fixkosten (invest_offset) nur wenn Kapazität > 0 – bei Kapazität 0 wird
    # die Komponente nicht gebaut, also entstehen auch keine Investitionskosten.
    def _capex(offset, spec_cost, cap):
        return (offset + spec_cost * cap) * annuity_factor if cap > 0 else 0.0

    def _invest(offset, spec_cost, cap):
        return (offset + spec_cost * cap) if cap > 0 else 0.0

    capex_hp       = _capex(hp_invest_offset,       hp_specific_cost,       hp_capacity)
    capex_storage  = _capex(storage_invest_offset,  storage_specific_cost,  storage_capacity_m3)
    capex_gas      = _capex(gas_invest_offset,       gas_specific_cost,      gas_boiler_capacity)
    capex_pv       = _capex(pv_invest_offset,        pv_specific_cost,       pv_capacity)
    capex_seasonal = _capex(seasonal_invest_offset,  seasonal_specific_cost, seasonal_capacity_m3)
    capex_grid     = (pipe_specific_cost * network_length) * annuity_factor

    # O&M: 1,5 % der Gesamtinvestition p.a.
    total_invest = (
        _invest(hp_invest_offset,       hp_specific_cost,       hp_capacity)
        + _invest(storage_invest_offset,  storage_specific_cost,  storage_capacity_m3)
        + _invest(gas_invest_offset,       gas_specific_cost,      gas_boiler_capacity)
        + _invest(pv_invest_offset,        pv_specific_cost,       pv_capacity)
        + _invest(seasonal_invest_offset,  seasonal_specific_cost, seasonal_capacity_m3)
        + pipe_specific_cost * network_length
    )
    opex_om = 0.015 * total_invest

    # --- OPEX (€/a) -----------------------------------------------------------
    opex_elec  = float(np.sum(electricity_price * P_grid))
    opex_pump  = float(P_pump * np.sum(electricity_price * (charge + discharge)))
    opex_gas   = float(np.sum(gas_price * (Q_gas_boiler / eta_gas_boiler)))

    # Netzentgelt-Leistungspreis (Fixkosten, €/a)
    _el = parameters["price_parameters"]["electricity"]
    _vbh = _el.get("vbh_class", "lower_2500VBH")
    capacity_charge = _el["network_charge"][_vbh]["capacity_charge"]   # €/kW*a
    opex_grid_capacity = capacity_charge * float(np.max(P_grid)) * 1000.0  # MW -> kW

    # --- PV-Erlös (Gutschrift, €/a) ------------------------------------------
    revenue_pv = float(np.sum(feed_in_tariff * pv_feed_in))

    # --- Gesamtkosten und LCOH -----------------------------------------------
    total_cost = (capex_hp + capex_storage + capex_gas + capex_pv + capex_seasonal
                  + capex_grid + opex_elec + opex_pump + opex_gas + opex_om + opex_grid_capacity - revenue_pv)

    heat_delivered = float(np.sum(demand))  # MWh/a
    lcoh_total = total_cost / heat_delivered

    # --- Komponenten-Dict (€/a, €/MWh, %-Anteil am Gesamt-LCOH) --------------
    raw = {
        "Wärmepumpe (CAPEX)":       capex_hp,
        "Pufferspeicher (CAPEX)":   capex_storage,
        "Gaskessel (CAPEX)":        capex_gas,
        "PV (CAPEX)":               capex_pv,
        "Saisonalspeicher (CAPEX)": capex_seasonal,
        "Wärmenetz (CAPEX)":        capex_grid,
        "Strombezug WP (OPEX)":     opex_elec,
        "Pumpstrom (OPEX)":         opex_pump,
        "Gas-Brennstoff (OPEX)":    opex_gas,
        "O&M (1,5 % CAPEX)":        opex_om,
        "PV-Einspeiseerlös":        -revenue_pv,   # negativ (Gutschrift)
        "Strombezug WP (OPEX)":     opex_elec,
        "Netzentgelt Leistungspreis": opex_grid_capacity,
        "Pumpstrom (OPEX)":         opex_pump,
    }

    components = {}
    for label, eur_a in raw.items():
        components[label] = {
            "eur_per_year": eur_a,
            "eur_per_mwh":  eur_a / heat_delivered,
            "share_pct":    eur_a / total_cost * 100.0,
        }

    if verbose:
        print("\n" + "=" * 64)
        print(f"{'LCOH-Aufschlüsselung':<32}{'€/a':>12}{'€/MWh':>10}{'Anteil':>8}")
        print("-" * 64)
        for label, v in components.items():
            print(f"{label:<32}{v['eur_per_year']:>12,.0f}{v['eur_per_mwh']:>10.2f}{v['share_pct']:>7.1f}%")
        print("-" * 64)
        print(f"{'Gelieferte Wärme':<32}{heat_delivered:>12,.0f} MWh/a")
        print(f"{'LCOH gesamt':<32}{total_cost:>12,.0f}{lcoh_total:>10.2f}{'100.0%':>8}")
        print("=" * 64 + "\n")

    return lcoh_total, components


_COLOR_WP       = "#00395B"
_COLOR_GAS      = "#C17A2F"
_COLOR_SPEICHER = "#769D7B"
_COLOR_SAISONAL = "#2F6B4F"
_COLOR_PV       = "#C8A84B"
_COLOR_NETZ     = "#A0463A"
_COLOR_PUMP     = "#888888"
_COLOR_OM       = "#AAAAAA"

_PIE_COLORS = {
    "Heat pump":              _COLOR_WP,
    "Buffer storage":         _COLOR_SPEICHER,
    "Gas boiler":             _COLOR_GAS,
    "PV (net)":               _COLOR_PV,
    "Seasonal storage":       _COLOR_SAISONAL,
    "District heating network": _COLOR_NETZ,
    "Pump electricity":       _COLOR_PUMP,
    "O&M (1.5 % CAPEX)":     _COLOR_OM,
}

_LABEL_EN = {
    "Wärmepumpe":       "Heat pump",
    "Pufferspeicher":   "Buffer storage",
    "Gaskessel":        "Gas boiler",
    "PV (netto)":       "PV (net)",
    "Saisonalspeicher": "Seasonal storage",
    "Wärmenetz":        "District heating network",
    "Pumpstrom":        "Pump electricity",
    "O&M":              "O&M (1.5 % CAPEX)",
}

_PLOTS_DIR = "src/ACES-2026/plots"


def plot_lcoh_pie(components, lcoh_total=None, show_plot=True):
    """
    Donut-Diagramm der LCOH-Zusammensetzung mit Projektfarben im PPT-Format.

    PV-CAPEX und PV-Erlös werden zu 'PV (netto)' zusammengefasst, damit der Donut
    nur positive Anteile enthält (negative Segmente sind in matplotlib nicht möglich).
    """
    pv_capex   = components.get("PV (CAPEX)", {}).get("eur_per_year", 0.0)
    pv_revenue = components.get("PV-Einspeiseerlös", {}).get("eur_per_year", 0.0)  # negativ
    pv_net     = pv_capex + pv_revenue

    pie_items = {
        "Wärmepumpe":       components["Wärmepumpe (CAPEX)"]["eur_per_year"]
                            + components["Strombezug WP (OPEX)"]["eur_per_year"],
        "Pufferspeicher":   components["Pufferspeicher (CAPEX)"]["eur_per_year"],
        "Gaskessel":        components["Gaskessel (CAPEX)"]["eur_per_year"]
                            + components["Gas-Brennstoff (OPEX)"]["eur_per_year"],
        "PV (netto)":       pv_net,
        "Saisonalspeicher": components["Saisonalspeicher (CAPEX)"]["eur_per_year"],
        "Wärmenetz":        components["Wärmenetz (CAPEX)"]["eur_per_year"],
        "Pumpstrom":        components["Pumpstrom (OPEX)"]["eur_per_year"],
        "O&M":              components["O&M (1,5 % CAPEX)"]["eur_per_year"],
    }

    total_positive = sum(v for v in pie_items.values() if v > 0)
    labels_en, values, colors = [], [], []
    for label, val in pie_items.items():
        en = _LABEL_EN.get(label, label)
        if val > 0:
            if val / total_positive * 100 < 0.1:
                print(f"Note: '{en}' is below 0.1 % ({val:,.0f} €/a) → not shown.")
                continue
            labels_en.append(en)
            values.append(val)
            colors.append(_PIE_COLORS.get(en, "#CCCCCC"))
        elif val != 0:
            print(f"Note: '{en}' is negative ({val:,.0f} €/a) → net credit, not shown.")

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Calibri', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")

    wedges, _, autotexts = ax.pie(
        values,
        labels=None,           # Labels über Legende, nicht direkt am Segment
        autopct=lambda p: f"{p:.1f} %",
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=1.18,      # außerhalb des Donuts
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
        t.set_color("#1A1A1A")

    if lcoh_total is not None:
        ax.text(0, 0, f"LCOH\n{lcoh_total:.1f} €/MWh",
                ha="center", va="center",
                fontsize=15, fontweight="bold", color="#1A1A1A")

    # Legende rechts
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=l) for l, c in zip(labels_en, colors)]
    ax.legend(handles=legend_handles, fontsize=13, frameon=False,
              loc="center left", bbox_to_anchor=(0.82, 0.5))

    ax.set_title("Levelized Cost of Heat (LCOH) – Component breakdown",
                 fontsize=16, fontweight="bold", color="#1A1A1A", pad=20)
    ax.axis("equal")

    os.makedirs(_PLOTS_DIR, exist_ok=True)
    out = os.path.join(_PLOTS_DIR, "lcoh_pie.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Plot gespeichert: {out}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return fig
