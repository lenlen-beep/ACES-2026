#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 2026

@author: matsbeyer

Vollständige techno-ökonomische Optimierung eines Fernwärmesystems
════════════════════════════════════════════════════════════════════

Technologien:
  ├── Wärmepumpe (WP)           – Grundlasterzeuger
  ├── Thermischer Pufferspeicher – zeitliche Flexibilität, Größe wird mitoptimiert
  ├── Gaskessel (GB)            – Spitzenlast / Backup
  └── Photovoltaik (PV)         – Eigenstromversorgung + Netzeinspeisung

Optimierungsziel:
  Minimierung der Gesamtbetriebskosten (Strom + Gas + annualisierte Speicherinvestition)

Wirtschaftliche Vollanalyse (post-Optimierung):
  Für jede Technologie: CAPEX, annualisierter CAPEX, OPEX, Betriebskosten
  System gesamt: LCOH, Gesamtinvestition, NPV, IRR, Amortisationszeit
  CO₂-Emissionen: Gaskessel, Netzstrom, Einsparung durch PV

Datengrundlage:
  - Fernwärmebedarf Flensburg 2017  (Excel)
  - PV-Ertrag Flensburg 2015        (CSV, renewables.ninja)
  - Börsenstrompreise DE/LU 2024    (Excel, SMARD)
"""

import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════════════════════
# 1) DATEIPFADE
# ═══════════════════════════════════════════════════════════════════════════════

load_file  = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/district_heating_data_Flensburg_2017.xlsx"
pv_file    = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/ninja_pv_54.7833_9.4333_uncorrected.csv"
price_file = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/Gro_handelspreise_202401010000_202501010000_Stunde.xlsx"

# ═══════════════════════════════════════════════════════════════════════════════
# 2) DATEN LADEN
# ═══════════════════════════════════════════════════════════════════════════════

df_load = pd.read_excel(load_file, skiprows=1, header=0)
df_load.columns = ['Datum', 'Wärmeleistung in MW']
df_load['Datum'] = pd.to_datetime(df_load['Datum'])

df_pv = pd.read_csv(pv_file, skiprows=3, header=0)
df_pv.columns = ['time', 'local_time', 'electricity', 'missing']
df_pv['electricity'] = pd.to_numeric(df_pv['electricity'], errors='coerce').fillna(0)

df_price = pd.read_excel(price_file, skiprows=9, header=0)
price_col = 'Deutschland/Luxemburg [€/MWh]'
df_price[price_col] = pd.to_numeric(df_price[price_col], errors='coerce').fillna(0)

n = min(len(df_load), len(df_pv), len(df_price))
T = range(n)

demand    = df_load['Wärmeleistung in MW'].values[:n]
pv_cf     = df_pv['electricity'].values[:n]
price_arr = df_price[price_col].values[:n]
price     = {t: float(price_arr[t]) for t in T}

# ═══════════════════════════════════════════════════════════════════════════════
# 3) TECHNOLOGIE-PARAMETER
# ═══════════════════════════════════════════════════════════════════════════════
# Jede Technologie hat:
#   - Betriebsparameter (Leistung, Wirkungsgrad, …)
#   - CAPEX  [€/MW oder €/MWh je nach Technologie]
#   - OPEX   [% des CAPEX p.a.]
#   - Lebensdauer und Zinssatz für Annuität

# ─── Allgemeine Finanzierungsparameter ───────────────────────────────────────
discount_rate    = 0.06   # Kalkulationszinssatz [-]
project_lifetime = 20     # Systembetrachtungshorizont [Jahre]

def annuity(capex, lifetime, rate):
    """Jährliche Kapitalkosten (Annuität) einer Investition."""
    af = rate * (1 + rate)**lifetime / ((1 + rate)**lifetime - 1)
    return capex * af

# ─── Wärmepumpe ──────────────────────────────────────────────────────────────
COP              = 3.5        # Leistungszahl [-]
P_wp_max         = 20         # Max. elektrische Leistung [MW_el]
#                             → max. Wärmeleistung = COP × P_wp_max

capex_wp_spec    = 1_200_000  # [€/MW_el] – Investitionskosten Großwärmepumpe
opex_wp_rate     = 0.02       # OPEX als Anteil CAPEX [%/a]
lifetime_wp      = 20         # Technische Lebensdauer [Jahre]
capex_wp         = P_wp_max * capex_wp_spec
opex_wp_annual   = opex_wp_rate * capex_wp
ann_capex_wp     = annuity(capex_wp, lifetime_wp, discount_rate)

# ─── Thermischer Pufferspeicher ──────────────────────────────────────────────
VLT              = 80         # Vorlauftemperatur [°C]
RLT              = 55         # Rücklauftemperatur [°C]
delta_T          = VLT - RLT  # Temperaturdifferenz [K]
cp_W             = 4.187      # Spez. Wärmekapazität Wasser [kJ/(kg·K)]
rho_W            = 1000       # Dichte Wasser [kg/m³]
vol_to_MWh       = rho_W * cp_W * delta_T / 3_600_000   # [MWh/m³]

max_charge_rate    = 0.25     # Max. Laderate [Anteil Kapazität/h]
max_discharge_rate = 0.25     # Max. Entladerate [Anteil Kapazität/h]
p_loss             = 1.67 / (24 * 1000)   # Thermische Verluste [MWh/h] – Technikkatalog

capex_sto_offset = 196_675    # Fixkostenanteil Speicher [€]
capex_sto_spec   = 86_139     # Variable Kosten [€/m³] (Technikkatalog, in €/L × 1000)
opex_sto_rate    = 0.02       # OPEX-Rate [%/a]
lifetime_sto     = 25         # Lebensdauer Speicher [Jahre]

n_sto_inv  = 20
r_sto_inv  = 0.05
af_sto     = r_sto_inv * (1 + r_sto_inv)**n_sto_inv / ((1 + r_sto_inv)**n_sto_inv - 1)

SOC_init_frac    = 0.0

# Pumpenergie-Penalty
g_pump    = 9.81; h_pump = 5; eta_p = 0.9
P_pump    = (g_pump * h_pump) / (eta_p * cp_W * 1000 * delta_T)

# ─── Gaskessel ───────────────────────────────────────────────────────────────
eta_gb           = 0.92       # Wirkungsgrad [-]
gas_price        = 40         # Gaspreis [€/MWh_th Brennstoff]
co2_gas          = 201        # CO₂-Emissionsfaktor Erdgas [kg CO₂/MWh_Brennstoff]

capex_gb_spec    = 120_000    # [€/MW_th] – Investitionskosten Gaskessel
opex_gb_rate     = 0.025      # OPEX-Rate [%/a]
lifetime_gb      = 25         # Lebensdauer [Jahre]
# CAPEX Gaskessel wird post-Optimierung berechnet (Größe = Spitzenlast aus Ergebnissen)

# ─── Photovoltaik ────────────────────────────────────────────────────────────
P_pv_peak        = 20         # Installierte Leistung [MW_el]
pv_avail         = pv_cf * P_pv_peak
feed_in_tariff   = 50         # Einspeisevergütung Direktvermarktung [€/MWh_el]
co2_grid         = 350        # CO₂-Emissionsfaktor Netzstrom [kg CO₂/MWh_el] – Bundesmix 2024

capex_pv_spec    = 700_000    # [€/MW_el] – Investitionskosten Utility-Scale PV
opex_pv_rate     = 0.015      # OPEX-Rate [%/a]
lifetime_pv      = 25         # Lebensdauer [Jahre]
capex_pv         = P_pv_peak * capex_pv_spec
opex_pv_annual   = opex_pv_rate * capex_pv
ann_capex_pv     = annuity(capex_pv, lifetime_pv, discount_rate)

# ═══════════════════════════════════════════════════════════════════════════════
# 4) VORAB-AUSGABE & CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

print("═" * 58)
print("  SYSTEMKONFIGURATION")
print("═" * 58)
print(f"  Spitzenlast Fernwärme:        {max(demand):>8.1f} MW_th")
print(f"  WP max. Wärmeleistung:        {COP * P_wp_max:>8.1f} MW_th  ({P_wp_max} MW_el)")
print(f"  Gaskessel:                    unbegrenzt (Backup)")
print(f"  PV installiert:               {P_pv_peak:>8.0f} MW_el")
print(f"  Ø Börsenstrompreis 2024:      {price_arr.mean():>8.1f} €/MWh")
print(f"  Einspeisevergütung PV:        {feed_in_tariff:>8.0f} €/MWh")
print("═" * 58)

# ═══════════════════════════════════════════════════════════════════════════════
# 5) HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def npv_func(rate, cashflows):
    return sum(cf / (1 + rate)**t for t, cf in enumerate(cashflows))

def compute_irr(cashflows):
    try:
        return brentq(lambda r: npv_func(r, cashflows), -0.99, 10.0)
    except ValueError:
        return None

def compute_payback(capex, net_savings_annual):
    return capex / net_savings_annual if net_savings_annual > 0 else float('inf')

# ═══════════════════════════════════════════════════════════════════════════════
# 6) PYOMO-OPTIMIERUNGSMODELL
# ═══════════════════════════════════════════════════════════════════════════════

model   = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)

# ─── Entscheidungsvariablen ──────────────────────────────────────────────────
model.P_wp             = pyo.Var(model.T, bounds=(0, P_wp_max))   # WP el. Leistung [MW_el]
model.charge           = pyo.Var(model.T, bounds=(0, None))        # Speicher laden [MW_th]
model.discharge        = pyo.Var(model.T, bounds=(0, None))        # Speicher entladen [MW_th]
model.SOC              = pyo.Var(model.T, bounds=(0, None))        # SOC [MWh_th]
model.storage_capacity = pyo.Var(bounds=(0, None), initialize=500) # Speichervolumen [m³]
model.P_grid           = pyo.Var(model.T, bounds=(0, None))        # Netzbezug [MW_el]
model.P_feed           = pyo.Var(model.T, bounds=(0, None))        # PV-Einspeisung [MW_el]
model.Q_gb             = pyo.Var(model.T, bounds=(0, None))        # Gaskessel [MW_th]

# ─── Zielfunktion ────────────────────────────────────────────────────────────
# Minimiere: Stromkosten + Pumpenergie + Gaskosten + annualisierte Speicherinvestition
def obj_rule(m):
    stromkosten    = sum(price[t] * m.P_grid[t]
                         - feed_in_tariff * m.P_feed[t] for t in m.T)
    pumpkosten     = P_pump * sum(price[t] * (m.charge[t] + m.discharge[t])
                                  for t in m.T)
    gaskosten      = sum((gas_price / eta_gb) * m.Q_gb[t] for t in m.T)
    speicherinvest = (capex_sto_offset
                      + capex_sto_spec * m.storage_capacity) * af_sto
    return stromkosten + pumpkosten + gaskosten + speicherinvest

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# ─── Constraints ─────────────────────────────────────────────────────────────

# Wärmebilanz
def heat_balance(m, t):
    return COP * m.P_wp[t] + m.discharge[t] + m.Q_gb[t] == demand[t] + m.charge[t]
model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Speicher-Dynamik mit thermischen Verlusten
def storage_dyn(m, t):
    cap_MWh = vol_to_MWh * m.storage_capacity
    if t == 0:
        return m.SOC[t] == SOC_init_frac * cap_MWh
    if t == T[-1]:
        return m.SOC[t] == SOC_init_frac * cap_MWh
    return m.SOC[t] == m.SOC[t-1] + m.charge[t] - m.discharge[t] - p_loss
model.storage_dyn = pyo.Constraint(model.T, rule=storage_dyn)

# SOC ≤ Speicherkapazität
def soc_cap(m, t):
    return m.SOC[t] <= vol_to_MWh * m.storage_capacity
model.soc_cap = pyo.Constraint(model.T, rule=soc_cap)

# Ladeleistung ≤ max_charge_rate × Kapazität
def charge_lim_sto(m, t):
    return m.charge[t] <= max_charge_rate * vol_to_MWh * m.storage_capacity
model.charge_lim_sto = pyo.Constraint(model.T, rule=charge_lim_sto)

# Ladeleistung ≤ WP-Maximalwärme
def charge_lim_wp(m, t):
    return m.charge[t] <= COP * P_wp_max
model.charge_lim_wp = pyo.Constraint(model.T, rule=charge_lim_wp)

# Entladeleistung ≤ max_discharge_rate × Kapazität
def discharge_lim(m, t):
    return m.discharge[t] <= max_discharge_rate * vol_to_MWh * m.storage_capacity
model.discharge_lim = pyo.Constraint(model.T, rule=discharge_lim)

# Kein Entladen bei negativen Strompreisen (Opportunitätskosten-Logik)
def neg_price(m, t):
    if price[t] < 0:
        return m.discharge[t] == 0
    return pyo.Constraint.Skip
model.neg_price = pyo.Constraint(model.T, rule=neg_price)

# Strombilanz: Netzbezug + PV = WP-Verbrauch + Einspeisung
def elec_balance(m, t):
    return m.P_grid[t] + pv_avail[t] == m.P_wp[t] + m.P_feed[t]
model.elec_balance = pyo.Constraint(model.T, rule=elec_balance)

# PV-Einspeisung ≤ verfügbares PV
def feed_lim(m, t):
    return m.P_feed[t] <= pv_avail[t]
model.feed_lim = pyo.Constraint(model.T, rule=feed_lim)

# ═══════════════════════════════════════════════════════════════════════════════
# 7) LÖSEN
# ═══════════════════════════════════════════════════════════════════════════════

print("\nOptimierung läuft ...")
solver  = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=False)
print(f"  Status: {results.solver.termination_condition}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 8) ERGEBNISSE EXTRAHIEREN
# ═══════════════════════════════════════════════════════════════════════════════

P_wp_res   = np.array([pyo.value(model.P_wp[t])      for t in T])
charge_res = np.array([pyo.value(model.charge[t])     for t in T])
disc_res   = np.array([pyo.value(model.discharge[t])  for t in T])
SOC_res    = np.array([pyo.value(model.SOC[t])        for t in T])
P_grid_res = np.array([pyo.value(model.P_grid[t])     for t in T])
P_feed_res = np.array([pyo.value(model.P_feed[t])     for t in T])
Q_gb_res   = np.array([pyo.value(model.Q_gb[t])       for t in T])
P_self_res = pv_avail - P_feed_res

storage_cap_m3  = pyo.value(model.storage_capacity)
storage_cap_MWh = vol_to_MWh * storage_cap_m3

demand = np.array(demand)

# ═══════════════════════════════════════════════════════════════════════════════
# 9) VOLLSTÄNDIGE TECHNO-ÖKONOMISCHE ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 9a) Energiemengen ───────────────────────────────────────────────────────
wp_heat_total   = (COP * P_wp_res).sum()          # [MWh_th/a]
gb_heat_total   = Q_gb_res.sum()                  # [MWh_th/a]
disc_total      = disc_res.sum()                  # [MWh_th/a]
charge_total    = charge_res.sum()                # [MWh_th/a]
demand_total    = demand.sum()                    # [MWh_th/a]
wp_el_total     = P_wp_res.sum()                  # [MWh_el/a] – WP-Strombedarf
grid_total      = P_grid_res.sum()                # [MWh_el/a]
pv_total        = pv_avail.sum()                  # [MWh_el/a]
pv_self_total   = P_self_res.sum()                # [MWh_el/a]
pv_feed_total   = P_feed_res.sum()                # [MWh_el/a]
gas_input_total = gb_heat_total / eta_gb          # [MWh_Brennstoff/a]

# ─── 9b) CO₂-Emissionen ──────────────────────────────────────────────────────
co2_gas_total   = gas_input_total * co2_gas / 1000          # [t CO₂/a]
co2_grid_total  = grid_total * co2_grid / 1000              # [t CO₂/a]
co2_pv_avoided  = pv_self_total * co2_grid / 1000           # [t CO₂/a] vermieden
co2_total       = co2_gas_total + co2_grid_total            # [t CO₂/a]
co2_specific    = co2_total * 1000 / demand_total           # [kg CO₂/MWh_th]

# ─── 9c) Betriebskosten ──────────────────────────────────────────────────────
strom_op        = price_arr @ P_grid_res                    # Netzbezugskosten [€/a]
feed_revenue    = feed_in_tariff * pv_feed_total            # Einspeisevergütung [€/a]
strom_netto     = strom_op - feed_revenue                   # Netto-Stromkosten [€/a]
gas_op          = (gas_price / eta_gb) * gb_heat_total      # Gaskosten [€/a]
pump_op         = P_pump * (price_arr @ (charge_res + disc_res))  # Pumpkosten [€/a]
total_op        = strom_netto + gas_op + pump_op            # Gesamtbetriebskosten [€/a]

# ─── 9d) CAPEX aller Technologien ────────────────────────────────────────────
# Gaskessel: CAPEX basiert auf tatsächlicher Spitzenlast aus Optimierung
P_gb_peak    = Q_gb_res.max()
capex_gb     = P_gb_peak * capex_gb_spec
opex_gb_annual = opex_gb_rate * capex_gb
ann_capex_gb = annuity(capex_gb, lifetime_gb, discount_rate)

# Speicher: CAPEX aus Optimierungsergebnis
capex_sto      = capex_sto_offset + capex_sto_spec * storage_cap_m3
opex_sto_annual = opex_sto_rate * capex_sto
ann_capex_sto  = annuity(capex_sto, lifetime_sto, discount_rate)

# Übersicht CAPEX
total_capex      = capex_wp + capex_sto + capex_gb + capex_pv

# ─── 9e) Gesamte jährliche Systemkosten ──────────────────────────────────────
# = Betriebskosten + OPEX + annualisierte CAPEX aller Technologien
ann_cost_wp      = ann_capex_wp  + opex_wp_annual
ann_cost_sto     = ann_capex_sto + opex_sto_annual
ann_cost_gb      = ann_capex_gb  + opex_gb_annual
ann_cost_pv      = ann_capex_pv  + opex_pv_annual - feed_revenue  # PV: Vergütung als Erlös
ann_cost_fuel    = gas_op + strom_op + pump_op                     # Energiekosten
total_annual_cost = ann_cost_fuel + (ann_capex_wp + opex_wp_annual
                                     + ann_capex_sto + opex_sto_annual
                                     + ann_capex_gb + opex_gb_annual
                                     + ann_capex_pv + opex_pv_annual
                                     - feed_revenue)

# Levelized Cost of Heat [€/MWh_th]
LCOH = total_annual_cost / demand_total

# ─── 9f) Systemweite NPV/IRR (Investition vs. Referenz ohne PV & Speicher) ──
# Referenz: nur Gaskessel ohne WP, PV, Speicher → Vollversorgung mit Gas
gas_ref_cost  = (gas_price / eta_gb) * demand_total   # [€/a] – reine Gaswärme als Referenz
savings_vs_ref = gas_ref_cost - total_annual_cost      # Einsparung ggü. Gas-only [€/a]
cashflows_sys  = [-total_capex] + [savings_vs_ref] * project_lifetime
npv_sys        = npv_func(discount_rate, cashflows_sys)
irr_sys        = compute_irr(cashflows_sys)
payback_sys    = compute_payback(total_capex, savings_vs_ref)

# ─── 9g) PV-Eigenanalyse (NPV nur PV-Investition) ────────────────────────────
# Einsparung PV = vermiedener Netzbezug (zu Spotpreisen) + Einspeisung
pv_savings_op  = (price_arr @ P_self_res) + feed_revenue   # operative Einsparung [€/a]
net_savings_pv = pv_savings_op - opex_pv_annual            # nach OPEX [€/a]
cashflows_pv   = [-capex_pv] + [net_savings_pv] * project_lifetime
npv_pv         = npv_func(discount_rate, cashflows_pv)
irr_pv         = compute_irr(cashflows_pv)
payback_pv     = compute_payback(capex_pv, net_savings_pv)

# ═══════════════════════════════════════════════════════════════════════════════
# 10) KONSOLENAUSGABE
# ═══════════════════════════════════════════════════════════════════════════════

SEP = "═" * 58

print(SEP)
print("  ENERGIEBILANZ")
print(SEP)
print(f"  Gesamtwärmebedarf:            {demand_total:>10.0f} MWh_th/a")
print(f"  Wärmepumpe (Wärme):           {wp_heat_total:>10.0f} MWh_th/a  ({wp_heat_total/demand_total*100:.1f} %)")
print(f"  Gaskessel:                    {gb_heat_total:>10.0f} MWh_th/a  ({gb_heat_total/demand_total*100:.1f} %)")
print(f"  Speicher (Entladung netto):   {disc_total-charge_total:>10.0f} MWh_th/a")
print(f"  Gaskessel Betriebsstunden:    {(Q_gb_res > 0.01).sum():>10} h/a")
print(f"  Gaskessel Spitzenlast:        {P_gb_peak:>10.1f} MW_th")
print()
print(f"  WP Strombedarf:               {wp_el_total:>10.0f} MWh_el/a")
print(f"  Netzbezug gesamt:             {grid_total:>10.0f} MWh_el/a")
print(f"  PV verfügbar:                 {pv_total:>10.0f} MWh_el/a")
print(f"  PV-Eigenverbrauch:            {pv_self_total:>10.0f} MWh_el/a  ({pv_self_total/pv_total*100:.1f} %)")
print(f"  PV-Einspeisung:               {pv_feed_total:>10.0f} MWh_el/a")
print(f"  Speichergröße (optimiert):    {storage_cap_m3:>10.0f} m³  =  {storage_cap_MWh:.1f} MWh_th")
print()
print(SEP)
print("  CO₂-EMISSIONEN")
print(SEP)
print(f"  Gas (Kessel):                 {co2_gas_total:>10.1f} t CO₂/a")
print(f"  Netzstrom (WP):               {co2_grid_total:>10.1f} t CO₂/a")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Gesamt:                       {co2_total:>10.1f} t CO₂/a")
print(f"  Spezifisch:                   {co2_specific:>10.1f} kg CO₂/MWh_th")
print(f"  Vermieden durch PV:           {co2_pv_avoided:>10.1f} t CO₂/a")
print()
print(SEP)
print("  CAPEX ALLER TECHNOLOGIEN")
print(SEP)
print(f"  {'Technologie':<30} {'CAPEX [€]':>12}  {'[Mio. €]':>9}")
print(f"  {'─'*52}")
print(f"  {'Wärmepumpe':<30} {capex_wp:>12,.0f}  {capex_wp/1e6:>9.2f}")
print(f"  {'Thermischer Speicher':<30} {capex_sto:>12,.0f}  {capex_sto/1e6:>9.2f}")
print(f"  {'Gaskessel':<30} {capex_gb:>12,.0f}  {capex_gb/1e6:>9.2f}")
print(f"  {'PV-Anlage':<30} {capex_pv:>12,.0f}  {capex_pv/1e6:>9.2f}")
print(f"  {'─'*52}")
print(f"  {'GESAMT':<30} {total_capex:>12,.0f}  {total_capex/1e6:>9.2f}")
print()
print(SEP)
print("  JÄHRLICHE KOSTEN (CAPEX ann. + OPEX + Energie)")
print(SEP)
print(f"  {'Technologie':<26} {'ann.CAPEX':>10} {'OPEX':>10} {'Energie':>10} {'Gesamt':>10}")
print(f"  {'─'*66}")
print(f"  {'Wärmepumpe':<26} {ann_capex_wp:>10,.0f} {opex_wp_annual:>10,.0f} {strom_netto:>10,.0f} {ann_capex_wp+opex_wp_annual+strom_netto:>10,.0f}")
print(f"  {'Thermischer Speicher':<26} {ann_capex_sto:>10,.0f} {opex_sto_annual:>10,.0f} {pump_op:>10,.0f} {ann_capex_sto+opex_sto_annual+pump_op:>10,.0f}")
print(f"  {'Gaskessel':<26} {ann_capex_gb:>10,.0f} {opex_gb_annual:>10,.0f} {gas_op:>10,.0f} {ann_capex_gb+opex_gb_annual+gas_op:>10,.0f}")
print(f"  {'PV (nach Einspeiseverg.)':<26} {ann_capex_pv:>10,.0f} {opex_pv_annual:>10,.0f} {-feed_revenue:>10,.0f} {ann_capex_pv+opex_pv_annual-feed_revenue:>10,.0f}")
print(f"  {'─'*66}")
print(f"  {'GESAMT':<26} {'':<10} {'':<10} {'':<10} {total_annual_cost:>10,.0f}")
print(f"  Alle Angaben in [€/a]")
print()
print(f"  LCOH (Wärmegestehungskosten): {LCOH:>10.2f} €/MWh_th")
print()
print(SEP)
print("  INVESTITIONSRECHNUNG GESAMTSYSTEM")
print("  (Referenz: vollständige Wärmeversorgung via Gaskessel)")
print(SEP)
print(f"  Gesamtinvestition:            {total_capex/1e6:>9.2f} Mio. €")
print(f"  Jährl. Einsparung ggü. Ref.:  {savings_vs_ref:>10,.0f} €/a")
print(f"  NPV ({discount_rate*100:.0f}%, {project_lifetime} a):              {npv_sys/1e6:>9.2f} Mio. €")
if irr_sys:
    print(f"  IRR Gesamtsystem:             {irr_sys*100:>9.1f} %")
else:
    print(f"  IRR Gesamtsystem:              nicht berechenbar")
print(f"  Amortisationszeit:            {payback_sys:>9.1f} Jahre")
print()
print(SEP)
print("  INVESTITIONSRECHNUNG PV-ANLAGE (isoliert)")
print(SEP)
print(f"  CAPEX:                        {capex_pv:>10,.0f} €")
print(f"  OPEX ({opex_pv_rate*100:.1f}% p.a.):              {opex_pv_annual:>10,.0f} €/a")
print(f"  Operative PV-Einsparung:      {pv_savings_op:>10,.0f} €/a")
print(f"  Netto-Einsparung:             {net_savings_pv:>10,.0f} €/a")
print(f"  NPV ({discount_rate*100:.0f}%, {project_lifetime} a):              {npv_pv/1e6:>9.2f} Mio. €")
if irr_pv:
    print(f"  IRR:                          {irr_pv*100:>9.1f} %")
else:
    print(f"  IRR:                           nicht berechenbar")
print(f"  Amortisationszeit:            {payback_pv:>9.1f} Jahre")
print(SEP)

# ═══════════════════════════════════════════════════════════════════════════════
# 11) PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

hpd    = 24
n_days = n // hpd
nd     = n_days * hpd
t_days = np.arange(n_days)

monatsgrenzen = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
monatsnamen   = ["Jan","Feb","Mär","Apr","Mai","Jun",
                 "Jul","Aug","Sep","Okt","Nov","Dez"]

def dm(arr):
    return arr[:nd].reshape(n_days, hpd).mean(axis=1)

demand_d   = dm(demand)
wp_heat_d  = dm(COP * P_wp_res)
gb_d       = dm(Q_gb_res)
disc_d     = dm(disc_res)
charge_d   = dm(charge_res)
SOC_d      = dm(SOC_res)
pv_avail_d = dm(pv_avail)
P_self_d   = dm(P_self_res)
P_feed_d   = dm(P_feed_res)
P_grid_d   = dm(P_grid_res)

# ── Plot 1: Jahresüberblick Wärmedeckung ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_days, demand_d,  label="Wärmebedarf",       color="black",      lw=2.0)
ax.plot(t_days, wp_heat_d, label="Wärmepumpe",        color="steelblue",  lw=1.5)
ax.plot(t_days, gb_d,      label="Gaskessel",          color="firebrick",  lw=1.5)
ax.plot(t_days, disc_d,    label="Speicher Entladung", color="darkorange", lw=1.2, ls="--")
ax.set_xticks(monatsgrenzen); ax.set_xticklabels(monatsnamen)
ax.set_ylabel("Ø Tagesleistung [MW_th]")
ax.set_title("Jahresüberblick: Wärmedeckung durch WP, Gaskessel und Speicher")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); plt.show()

# ── Plot 2: Speicher SOC + Laden/Entladen ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax1.plot(t_days, charge_d, label="Laden [MW_th]",    color="royalblue")
ax1.plot(t_days, disc_d,   label="Entladen [MW_th]", color="tomato")
ax1.set_ylabel("Ø Tagesleistung [MW_th]")
ax1.set_title(f"Thermischer Speicher: Betrieb und SOC  "
              f"(opt. Größe: {storage_cap_m3:.0f} m³ = {storage_cap_MWh:.0f} MWh_th)")
ax1.legend(); ax1.grid(alpha=0.4)

ax2.plot(t_days, SOC_d, color="seagreen", label="SOC [MWh_th]")
ax2.axhline(storage_cap_MWh, color="gray", ls="--",
            label=f"Kapazität {storage_cap_MWh:.0f} MWh_th")
ax2.set_xticks(monatsgrenzen); ax2.set_xticklabels(monatsnamen)
ax2.set_xlabel("Monat")
ax2.set_ylabel("Ø Tages-SOC [MWh_th]")
ax2.legend(); ax2.grid(alpha=0.4)
plt.tight_layout(); plt.show()

# ── Plot 3: PV-Strombilanz Jahresüberblick ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_days, pv_avail_d, label="PV verfügbar [MW_el]",       color="orange",    lw=1.5)
ax.plot(t_days, P_self_d,   label="PV-Eigenverbrauch [MW_el]",  color="gold",      lw=1.3)
ax.plot(t_days, P_feed_d,   label="PV-Einspeisung [MW_el]",     color="limegreen", lw=1.3)
ax.plot(t_days, P_grid_d,   label="Netzbezug WP [MW_el]",       color="steelblue", lw=1.3)
ax.set_xticks(monatsgrenzen); ax.set_xticklabels(monatsnamen)
ax.set_xlabel("Monat")
ax.set_ylabel("Ø Tagesleistung [MW_el]")
ax.set_title("Jahresüberblick: PV-Strombilanz (Tagesmittelwerte)")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); plt.show()

# ── Plot 4: Dauerlinie ────────────────────────────────────────────────────────
sidx = np.argsort(-demand)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(demand[sidx],              label="Wärmebedarf",       color="black",      lw=2.0)
ax.plot((COP * P_wp_res)[sidx],   label="Wärmepumpe",        color="steelblue",  lw=1.5)
ax.plot(Q_gb_res[sidx],           label="Gaskessel",          color="firebrick",  lw=1.5, ls="--")
ax.plot(disc_res[sidx],           label="Speicher Entladung", color="darkorange", lw=1.2, ls=":")
ax.set_xlabel("Stunden (sortiert nach Wärmebedarf)")
ax.set_ylabel("Leistung [MW_th]")
ax.set_title("Dauerlinie: Wärmeerzeugung")
ax.legend(); ax.grid(alpha=0.4)
plt.tight_layout(); plt.show()

# ── Plot 5: Kostenstruktur – gestapeltes Balkendiagramm ──────────────────────
technologien = ["Wärmepumpe", "Speicher", "Gaskessel", "PV"]
kosten_kapex  = [ann_capex_wp,  ann_capex_sto,  ann_capex_gb,  ann_capex_pv]
kosten_opex   = [opex_wp_annual, opex_sto_annual, opex_gb_annual, opex_pv_annual]
kosten_energie= [strom_netto,   pump_op,        gas_op,        -feed_revenue]
x = np.arange(len(technologien))

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x, np.array(kosten_kapex)/1e6,  label="Annualisierter CAPEX",
            color="steelblue", alpha=0.85)
b2 = ax.bar(x, np.array(kosten_opex)/1e6,   bottom=np.array(kosten_kapex)/1e6,
            label="OPEX", color="darkorange", alpha=0.85)
b3 = ax.bar(x, np.array(kosten_energie)/1e6,
            bottom=(np.array(kosten_kapex)+np.array(kosten_opex))/1e6,
            label="Energiekosten (netto)", color="seagreen", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(technologien)
ax.set_ylabel("Jährliche Kosten [Mio. €/a]")
ax.set_title(f"Kostenstruktur aller Technologien  |  LCOH = {LCOH:.1f} €/MWh_th")
ax.legend(); ax.grid(axis="y", alpha=0.4)
ax.axhline(0, color="black", lw=0.8)
plt.tight_layout(); plt.show()

# ── Plot 6: CAPEX-Aufteilung Tortendiagramm ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
capex_vals   = [capex_wp, capex_sto, capex_gb, capex_pv]
capex_labels = [f"WP\n{capex_wp/1e6:.1f} Mio. €",
                f"Speicher\n{capex_sto/1e6:.1f} Mio. €",
                f"Gaskessel\n{capex_gb/1e6:.1f} Mio. €",
                f"PV\n{capex_pv/1e6:.1f} Mio. €"]
colors = ["steelblue", "seagreen", "firebrick", "gold"]
ax.pie(capex_vals, labels=capex_labels, colors=colors, autopct="%1.1f%%",
       startangle=140, pctdistance=0.75)
ax.set_title(f"CAPEX-Aufteilung  |  Gesamt: {total_capex/1e6:.1f} Mio. €")
plt.tight_layout(); plt.show()

# ── Plot 7: System-NPV-Verlauf über Projektlaufzeit ──────────────────────────
years       = np.arange(project_lifetime + 1)
cum_cf      = np.array([sum(cashflows_sys[:y+1]) for y in range(len(cashflows_sys))])
cum_npv_arr = np.array([npv_func(discount_rate, cashflows_sys[:y+1])
                         for y in range(len(cashflows_sys))])

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(years, cum_cf / 1e6, label="Kumulierter Cashflow (nominal)", alpha=0.6, color="steelblue")
ax.plot(years, cum_npv_arr / 1e6, color="crimson", lw=2,
        label="Kumulierter NPV (diskontiert)")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Jahr")
ax.set_ylabel("Mio. €")
ax.set_title(f"System-Investitionsrechnung  (Referenz: Gas-only)  |  "
             f"Amortisation: {payback_sys:.1f} a")
ax.legend(); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); plt.show()
