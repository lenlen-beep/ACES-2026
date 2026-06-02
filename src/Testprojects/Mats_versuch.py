#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 2026

@author: matsbeyer

Kombiniertes Fernwärmesystem: Wärmepumpe + Thermischer Speicher + Gaskessel + Photovoltaik

Dieses Skript vereint alle Komponenten aus den Einzelskripten des Testprojekts:
  - Simple_energy_system.py  → Grundstruktur WP + Speicher
  - Storage_optimization.py  → Physikalisches Speichermodell (Volumen, Verluste, Investition)
  - Gas_boiler_integration.py → Gaskessel als Spitzenlast-/Backuperzeuger
  - Mats.py                  → PV-Anlage, echte Börsenstrompreise, Wirtschaftsanalyse

Optimierungsziel:
  Minimierung der Gesamtbetriebskosten (Strom + Gas) und der annualisierten
  Speicherinvestition über einen Jahreshorizont.

Entscheidungsvariablen:
  - P_wp[t]          : elektrische Leistung Wärmepumpe [MW_el]
  - charge[t]        : Ladeleistung Wärmespeicher [MW_th]
  - discharge[t]     : Entladeleistung Wärmespeicher [MW_th]
  - SOC[t]           : Ladezustand Wärmespeicher [MWh_th]
  - storage_capacity : Speichervolumen [m³] – wird mitoptimiert
  - P_grid[t]        : Netzbezug Strom [MW_el]
  - P_feed[t]        : PV-Einspeisung ins Netz [MW_el]
  - Q_gb[t]          : Wärmeleistung Gaskessel [MW_th]

Nebenbedingungen:
  Wärmebilanz, Speicherdynamik, Speicherkapazität, Leistungsgrenzen, Strombilanz

Datengrundlage:
  - Fernwärmebedarf Flensburg 2017 (Excel)
  - PV-Ertrag Flensburg 2015 (CSV, renewables.ninja)
  - Börsenstrompreise DE/LU 2024 (Excel, SMARD)
"""

import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# =============================================================================
# 1) DATEIPFADE
# =============================================================================

load_file  = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/district_heating_data_Flensburg_2017.xlsx"
pv_file    = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/ninja_pv_54.7833_9.4333_uncorrected.csv"
price_file = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/Gro_handelspreise_202401010000_202501010000_Stunde.xlsx"

# =============================================================================
# 2) DATEN LADEN
# =============================================================================

# --- Fernwärmelast [MW_th] ---
df_load = pd.read_excel(load_file, skiprows=1, header=0)
df_load.columns = ['Datum', 'Wärmeleistung in MW']
df_load['Datum'] = pd.to_datetime(df_load['Datum'])

# --- PV-Ertrag (renewables.ninja, Flensburg 2015) ---
# Spalte 'electricity': normierter Ertrag [kW/kW_peak] → Kapazitätsfaktor [-]
df_pv = pd.read_csv(pv_file, skiprows=3, header=0)
df_pv.columns = ['time', 'local_time', 'electricity', 'missing']
df_pv['electricity'] = pd.to_numeric(df_pv['electricity'], errors='coerce').fillna(0)

# --- Börsenstrompreise DE/LU 2024 (SMARD) [€/MWh_el] ---
df_price = pd.read_excel(price_file, skiprows=9, header=0)
price_col = 'Deutschland/Luxemburg [€/MWh]'
df_price[price_col] = pd.to_numeric(df_price[price_col], errors='coerce').fillna(0)

# Alle Reihen auf gemeinsame Länge kürzen (kürzeste Datei bestimmt n)
n = min(len(df_load), len(df_pv), len(df_price))
T = range(n)

demand    = df_load['Wärmeleistung in MW'].values[:n]   # [MW_th]
pv_cf     = df_pv['electricity'].values[:n]             # [-] Kapazitätsfaktor
price_arr = df_price[price_col].values[:n]              # [€/MWh_el]
price     = {t: float(price_arr[t]) for t in T}

# =============================================================================
# 3) PARAMETER
# =============================================================================

# --- Wärmepumpe ---
COP      = 3.5    # Leistungszahl [-]
P_wp_max = 20     # Max. elektrische Leistung [MW_el]
#   → max. Wärmeleistung WP = COP * P_wp_max

# --- Thermischer Pufferspeicher (physikalisches Modell) ---
# Netztemperaturen
VLT      = 80     # Vorlauftemperatur [°C]
RLT      = 55     # Rücklauftemperatur [°C]
delta_T  = VLT - RLT                   # Temperaturdifferenz [K]
cp_W     = 4.187  # spez. Wärmekapazität Wasser [kJ/(kg·K)]
rho_W    = 1000   # Dichte Wasser [kg/m³]

# Umrechnungsfaktor: 1 m³ Speicher → MWh_th
# E [MWh] = V [m³] * rho [kg/m³] * cp [kJ/(kg·K)] * ΔT [K] / 3 600 000
vol_to_MWh = rho_W * cp_W * delta_T / 3_600_000   # [MWh/m³]

# Speicher-Betriebsparameter
max_charge_rate    = 0.25    # max. Laderate als Anteil der Kapazität [1/h]
max_discharge_rate = 0.25    # max. Entladerate als Anteil der Kapazität [1/h]
p_loss             = 1.67 / (24 * 1000)  # thermische Verlustleistung [MWh/h] – Technikkatalog

# Investitionskosten Speicher (linearisiert, aus Technikkatalog)
# Annuität: 20 Jahre, 5 % Zins
n_inv            = 20
r_inv            = 0.05
annuity_factor   = r_inv * (1 + r_inv)**n_inv / ((1 + r_inv)**n_inv - 1)
storage_offset   = 196_675         # Fixkosten [€]
storage_specific = 86.139 / 1000   # variable Kosten [€/L] = [€/m³ * 1000]
# Hinweis: storage_specific ist €/Liter → Umrechnung auf €/m³ * 0.001 (da 1 m³ = 1000 L)
# Tatsächlich: 86.139 €/m³ (da der Wert aus Technikkatalog in €/L angegeben, * 1000 → €/m³)
storage_specific_m3 = storage_specific * 1000   # [€/m³]

# Pumpenergie-Penalty (aus Storage_optimization.py)
g       = 9.81    # [m/s²]
h_pump  = 5       # Förderhöhe [m]
eta_p   = 0.9     # Pumpenwirkungsgrad [-]
P_pump  = (g * h_pump) / (eta_p * cp_W * 1000 * delta_T)  # [MW_el / MW_th Durchsatz]

# SOC-Randbedingungen
SOC_init_frac = 0.0   # Speicher startet und endet leer

# --- Gaskessel (Backup/Spitzenlast) ---
eta_gb    = 0.92   # Kesselwirkungsgrad [-]
gas_price = 40     # Gaspreis [€/MWh_th Brennstoff]
# Kein P_gb_max → Gaskessel liefert unbegrenzt, was WP+Speicher nicht schafft

# --- Solarpark (PV) ---
P_pv_peak      = 10     # Installierte Leistung [MW_el]
pv_avail       = pv_cf * P_pv_peak   # verfügbare PV-Leistung je Stunde [MW_el]
feed_in_tariff = 40     # Einspeisevergütung Direktvermarktung [€/MWh_el]
# Muss < Ø-Spotpreis, damit Eigenverbrauch attraktiver als Einspeisung

# --- Wirtschaftliche Parameter PV (Post-Optimierung) ---
specific_capex_pv = 700_000   # [€/MW_el installiert]
opex_rate         = 0.015     # jährl. Betriebskosten als Anteil CAPEX [-]
project_lifetime  = 20        # [Jahre]
discount_rate     = 0.06      # Kalkulationszinssatz [-]

# --- Vorab-Check: Ist P_wp_max ausreichend? ---
# Mit Gaskessel als Backup immer lösbar, aber Info trotzdem sinnvoll
P_wp_min = max(0, (max(demand)) / COP)  # ohne Speicher und ohne Gas
print("=" * 55)
print("  SYSTEMKONFIGURATION")
print("=" * 55)
print(f"  Spitzenlast Fernwärme:      {max(demand):>8.1f} MW_th")
print(f"  WP Max. Wärmeleistung:      {COP * P_wp_max:>8.1f} MW_th")
print(f"  Gasbacker: vorhanden (unbegrenzt)")
print(f"  PV installiert:             {P_pv_peak:>8.0f} MW_el")
print(f"  Ø Börsenstrompreis:         {price_arr.mean():>8.1f} €/MWh")
print(f"  Einspeisevergütung:         {feed_in_tariff:>8.0f} €/MWh")
print("=" * 55)

# =============================================================================
# 4) HILFSFUNKTIONEN
# =============================================================================

def npv_func(rate, cashflows):
    return sum(cf / (1 + rate)**t for t, cf in enumerate(cashflows))

# =============================================================================
# 5) PYOMO-MODELL
# =============================================================================

model   = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)

# --- 5.1 Entscheidungsvariablen ---

# Wärmepumpe
model.P_wp      = pyo.Var(model.T, bounds=(0, P_wp_max))   # [MW_el]

# Thermischer Speicher
model.charge    = pyo.Var(model.T, bounds=(0, None))        # [MW_th]
model.discharge = pyo.Var(model.T, bounds=(0, None))        # [MW_th]
model.SOC       = pyo.Var(model.T, bounds=(0, None))        # [MWh_th]
model.storage_capacity = pyo.Var(bounds=(0, None),
                                 initialize=500)             # [m³] – wird optimiert

# Strombilanz (PV)
model.P_grid    = pyo.Var(model.T, bounds=(0, None))        # Netzbezug [MW_el]
model.P_feed    = pyo.Var(model.T, bounds=(0, None))        # PV-Einspeisung [MW_el]

# Gaskessel
model.Q_gb      = pyo.Var(model.T, bounds=(0, None))        # [MW_th]

# --- 5.2 Zielfunktion ---

def obj_rule(m):
    # Stromkosten (Netzbezug minus Einspeisevergütung)
    stromkosten  = sum(price[t] * m.P_grid[t]
                       - feed_in_tariff * m.P_feed[t]
                       for t in m.T)

    # Pumpenergie-Kosten (proportional zu Lade-/Entladedurchsatz)
    pumpkosten   = P_pump * sum(price[t] * (m.charge[t] + m.discharge[t])
                                for t in m.T)

    # Gaskosten
    gaskosten    = sum((gas_price / eta_gb) * m.Q_gb[t] for t in m.T)

    # Annualisierte Speicherinvestition
    speicherinvest = (storage_offset
                      + storage_specific_m3 * m.storage_capacity) * annuity_factor

    return stromkosten + pumpkosten + gaskosten + speicherinvest

model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# --- 5.3 Constraints ---

# Wärmebilanz: WP + Speicher-Entladung + Gas = Nachfrage + Speicher-Ladung
def heat_balance(m, t):
    return COP * m.P_wp[t] + m.discharge[t] + m.Q_gb[t] == demand[t] + m.charge[t]

model.heat_balance = pyo.Constraint(model.T, rule=heat_balance)

# Speicher-Dynamik mit thermischen Verlusten
def storage_rule(m, t):
    cap_MWh = vol_to_MWh * m.storage_capacity   # Kapazität in MWh (linear in Variable)
    if t == 0:
        return m.SOC[t] == SOC_init_frac * cap_MWh
    if t == T[-1]:
        return m.SOC[t] == SOC_init_frac * cap_MWh
    return m.SOC[t] == m.SOC[t-1] + m.charge[t] - m.discharge[t] - p_loss

model.storage_dyn = pyo.Constraint(model.T, rule=storage_rule)

# SOC darf Kapazität nicht überschreiten
def soc_cap_rule(m, t):
    return m.SOC[t] <= vol_to_MWh * m.storage_capacity

model.soc_cap = pyo.Constraint(model.T, rule=soc_cap_rule)

# Ladeleistung ≤ max_charge_rate × Kapazität (Speicher-Grenze)
def charge_limit_storage(m, t):
    return m.charge[t] <= max_charge_rate * vol_to_MWh * m.storage_capacity

model.charge_lim_storage = pyo.Constraint(model.T, rule=charge_limit_storage)

# Ladeleistung ≤ WP-Wärmeleistung (kann nicht mehr laden als WP liefert)
def charge_limit_wp(m, t):
    return m.charge[t] <= COP * P_wp_max

model.charge_lim_wp = pyo.Constraint(model.T, rule=charge_limit_wp)

# Entladeleistung ≤ max_discharge_rate × Kapazität
def discharge_limit(m, t):
    return m.discharge[t] <= max_discharge_rate * vol_to_MWh * m.storage_capacity

model.discharge_lim = pyo.Constraint(model.T, rule=discharge_limit)

# Kein Entladen bei negativen Strompreisen
# (billiger, die WP direkt zu betreiben als gespeicherte Wärme zu nutzen)
def neg_price_rule(m, t):
    if price[t] < 0:
        return m.discharge[t] == 0
    return pyo.Constraint.Skip

model.neg_price_con = pyo.Constraint(model.T, rule=neg_price_rule)

# Strombilanz: Netzbezug + PV = WP-Verbrauch + Einspeisung
def elec_balance(m, t):
    return m.P_grid[t] + pv_avail[t] == m.P_wp[t] + m.P_feed[t]

model.elec_balance = pyo.Constraint(model.T, rule=elec_balance)

# PV-Einspeisung ≤ verfügbares PV (kein Netzbezug zum Einspeisen)
def feed_limit(m, t):
    return m.P_feed[t] <= pv_avail[t]

model.feed_lim = pyo.Constraint(model.T, rule=feed_limit)

# =============================================================================
# 6) SOLVER
# =============================================================================

print("\nOptimierung läuft ...")
solver  = pyo.SolverFactory('glpk')
results = solver.solve(model, tee=False)
print(f"  Solver-Status: {results.solver.termination_condition}")

# =============================================================================
# 7) ERGEBNISSE EXTRAHIEREN
# =============================================================================

P_wp_res   = np.array([pyo.value(model.P_wp[t])      for t in T])
charge_res = np.array([pyo.value(model.charge[t])     for t in T])
disc_res   = np.array([pyo.value(model.discharge[t])  for t in T])
SOC_res    = np.array([pyo.value(model.SOC[t])        for t in T])
P_grid_res = np.array([pyo.value(model.P_grid[t])     for t in T])
P_feed_res = np.array([pyo.value(model.P_feed[t])     for t in T])
Q_gb_res   = np.array([pyo.value(model.Q_gb[t])       for t in T])
P_self_res = pv_avail - P_feed_res   # PV-Eigenverbrauch [MW_el]

storage_cap_m3  = pyo.value(model.storage_capacity)
storage_cap_MWh = vol_to_MWh * storage_cap_m3

demand = np.array(demand)

# =============================================================================
# 8) WIRTSCHAFTLICHE KENNZAHLEN
# =============================================================================

# --- Betriebskosten ---
stromkosten_op   = price_arr @ P_grid_res - feed_in_tariff * P_feed_res.sum()
gaskosten_op     = (gas_price / eta_gb) * Q_gb_res.sum()
speicher_invest  = (storage_offset + storage_specific_m3 * storage_cap_m3) * annuity_factor
gesamtkosten_op  = stromkosten_op + gaskosten_op + speicher_invest

# Referenz ohne PV (gleicher Dispatch, PV = 0)
stromkosten_nopv = price_arr @ P_wp_res   # nur WP-Netzbezug
pv_einsparung    = (stromkosten_nopv - stromkosten_op
                    + feed_in_tariff * P_feed_res.sum())  # vermiedene Kosten + Vergütung

# Spezifische Wärmekosten
total_heat_MWh   = (COP * P_wp_res + Q_gb_res).sum()
spez_waermekosten = gesamtkosten_op / total_heat_MWh

# --- PV-Investitionsrechnung ---
capex_pv          = P_pv_peak * specific_capex_pv
opex_pv_annual    = opex_rate * capex_pv
net_savings_pv    = pv_einsparung - opex_pv_annual
cashflows_pv      = [-capex_pv] + [net_savings_pv] * project_lifetime
npv_pv            = npv_func(discount_rate, cashflows_pv)

try:
    irr_pv = brentq(lambda r: npv_func(r, cashflows_pv), -0.99, 10.0)
except ValueError:
    irr_pv = None

payback_pv = capex_pv / net_savings_pv if net_savings_pv > 0 else float('inf')

# =============================================================================
# 9) KONSOLENAUSGABE
# =============================================================================

print()
print("=" * 55)
print("  THERMISCHER SPEICHER (optimiert)")
print("=" * 55)
print(f"  Optimale Speichergröße:     {storage_cap_m3:>8.0f} m³")
print(f"  Entspricht:                 {storage_cap_MWh:>8.1f} MWh_th")
print(f"  Volllaststunden:            {(SOC_res > 0.01 * storage_cap_MWh).sum():>8} h")
print(f"  Speicher-Lade-Energie:      {charge_res.sum():>8.0f} MWh_th/a")
print(f"  Speicher-Entlade-Energie:   {disc_res.sum():>8.0f} MWh_th/a")
print(f"  Annualisierte Investition:  {speicher_invest:>8.0f} €/a")
print()
print("=" * 55)
print("  WÄRMEERZEUGUNG")
print("=" * 55)
wp_heat_sum = (COP * P_wp_res).sum()
gb_heat_sum = Q_gb_res.sum()
total_heat  = wp_heat_sum + disc_res.sum()
print(f"  Gesamtwärmebedarf:          {demand.sum():>8.0f} MWh_th/a")
print(f"  Wärmepumpe (Wärme):         {wp_heat_sum:>8.0f} MWh_th/a"
      f"  ({wp_heat_sum/demand.sum()*100:.1f} %)")
print(f"  Gaskessel:                  {gb_heat_sum:>8.0f} MWh_th/a"
      f"  ({gb_heat_sum/demand.sum()*100:.1f} %)")
print(f"  Gaskessel Betriebsstunden:  {(Q_gb_res > 0.01).sum():>8} h/a")
print(f"  Gaskessel Spitzenlast:      {Q_gb_res.max():>8.1f} MW_th")
print()
print("=" * 55)
print("  SOLARPARK PV")
print("=" * 55)
print(f"  PV verfügbar gesamt:        {pv_avail.sum():>8.0f} MWh_el/a")
print(f"  PV-Eigenverbrauch:          {P_self_res.sum():>8.0f} MWh_el/a")
print(f"  PV-Einspeisung:             {P_feed_res.sum():>8.0f} MWh_el/a")
print(f"  Eigenverbrauchsquote:       {P_self_res.sum()/pv_avail.sum()*100:>7.1f} %")
print(f"  WP-Netzbezug gesamt:        {P_grid_res.sum():>8.0f} MWh_el/a")
print()
print("=" * 55)
print("  BETRIEBSKOSTEN")
print("=" * 55)
print(f"  Stromkosten (netto):        {stromkosten_op:>10.0f} €/a")
print(f"  Gaskosten:                  {gaskosten_op:>10.0f} €/a")
print(f"  Speicher-Invest (ann.):     {speicher_invest:>10.0f} €/a")
print(f"  ──────────────────────────────────────────────────")
print(f"  Gesamtkosten:               {gesamtkosten_op:>10.0f} €/a")
print(f"  Spez. Wärmekosten:          {spez_waermekosten:>10.2f} €/MWh_th")
print()
print("=" * 55)
print("  PV-INVESTITIONSRECHNUNG")
print("=" * 55)
print(f"  CAPEX ({P_pv_peak:.0f} MW × {specific_capex_pv/1e3:.0f} k€/MW):  {capex_pv:>10.0f} €")
print(f"  OPEX ({opex_rate*100:.1f}% p.a.):              {opex_pv_annual:>10.0f} €/a")
print(f"  Jährl. PV-Einsparung:       {pv_einsparung:>10.0f} €/a")
print(f"  Netto-Einsparung:           {net_savings_pv:>10.0f} €/a")
print(f"  NPV ({discount_rate*100:.0f}%, {project_lifetime} a):            {npv_pv:>10.0f} €")
if irr_pv is not None:
    print(f"  IRR:                        {irr_pv*100:>9.1f} %")
else:
    print(f"  IRR:                         nicht berechenbar")
print(f"  Amortisationszeit:          {payback_pv:>9.1f} Jahre")
print("=" * 55)

# =============================================================================
# 10) PLOTS
# =============================================================================

# Hilfsgrössen für Tagesmittelwerte
hpd    = 24             # Stunden pro Tag
n_days = n // hpd
nd     = n_days * hpd   # abschneiden auf ganze Tage

monatsgrenzen = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
monatsnamen   = ["Jan","Feb","Mär","Apr","Mai","Jun",
                 "Jul","Aug","Sep","Okt","Nov","Dez"]
t_days = np.arange(n_days)

def daily_mean(arr):
    return arr[:nd].reshape(n_days, hpd).mean(axis=1)

demand_d   = daily_mean(demand)
wp_heat_d  = daily_mean(COP * P_wp_res)
gb_d       = daily_mean(Q_gb_res)
disc_d     = daily_mean(disc_res)
charge_d   = daily_mean(charge_res)
SOC_d      = daily_mean(SOC_res)
pv_avail_d = daily_mean(pv_avail)
P_self_d   = daily_mean(P_self_res)
P_feed_d   = daily_mean(P_feed_res)
P_grid_d   = daily_mean(P_grid_res)

# ── Plot 1: Jahresüberblick Wärmedeckung ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_days, demand_d,  label="Wärmebedarf [MW_th]",
        color="black",      linewidth=2.0)
ax.plot(t_days, wp_heat_d, label="Wärmepumpe [MW_th]",
        color="steelblue",  linewidth=1.5)
ax.plot(t_days, gb_d,      label="Gaskessel [MW_th]",
        color="firebrick",  linewidth=1.5)
ax.plot(t_days, disc_d,    label="Speicher Entladung [MW_th]",
        color="darkorange", linewidth=1.2, linestyle="--")
ax.set_xticks(monatsgrenzen); ax.set_xticklabels(monatsnamen)
ax.set_xlabel("Monat")
ax.set_ylabel("Ø Tagesleistung [MW_th]")
ax.set_title("Jahresüberblick: Wärmedeckung durch WP, Gas und Speicher")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); plt.show()

# ── Plot 2: Speicher – SOC, Laden und Entladen ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1.plot(t_days, charge_d, label="Laden [MW_th]",    color="royalblue")
ax1.plot(t_days, disc_d,   label="Entladen [MW_th]",  color="tomato")
ax1.set_ylabel("Ø Tagesleistung [MW_th]")
ax1.set_title("Thermischer Speicher: Lade-/Entladevorgänge und SOC")
ax1.legend(); ax1.grid(alpha=0.4)

ax2.plot(t_days, SOC_d,   color="seagreen", label="SOC [MWh_th]")
ax2.axhline(storage_cap_MWh, color="gray", linestyle="--",
            label=f"Kapazität {storage_cap_MWh:.0f} MWh")
ax2.set_xticks(monatsgrenzen); ax2.set_xticklabels(monatsnamen)
ax2.set_xlabel("Monat")
ax2.set_ylabel("Ø Tages-SOC [MWh_th]")
ax2.legend(); ax2.grid(alpha=0.4)

plt.tight_layout(); plt.show()

# ── Plot 3: Jahresüberblick PV-Strombilanz ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_days, pv_avail_d, label="PV verfügbar [MW_el]",
        color="orange",     linewidth=1.5)
ax.plot(t_days, P_self_d,   label="PV-Eigenverbrauch [MW_el]",
        color="gold",       linewidth=1.3)
ax.plot(t_days, P_feed_d,   label="Einspeisung [MW_el]",
        color="limegreen",  linewidth=1.3)
ax.plot(t_days, P_grid_d,   label="Netzbezug WP [MW_el]",
        color="steelblue",  linewidth=1.3)
ax.set_xticks(monatsgrenzen); ax.set_xticklabels(monatsnamen)
ax.set_xlabel("Monat")
ax.set_ylabel("Ø Tagesleistung [MW_el]")
ax.set_title("Jahresüberblick: PV-Strombilanz (Tagesmittelwerte)")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); plt.show()

# ── Plot 4: Dauerlinie (sortiert nach Wärmebedarf) ────────────────────────────
sorted_idx    = np.argsort(-demand)
demand_s      = demand[sorted_idx]
wp_heat_s     = (COP * P_wp_res)[sorted_idx]
gb_s          = Q_gb_res[sorted_idx]
disc_s        = disc_res[sorted_idx]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(demand_s,  label="Wärmebedarf",            color="black",      linewidth=2)
ax.plot(wp_heat_s, label="Wärmepumpe (Wärme)",     color="steelblue",  linewidth=1.5)
ax.plot(gb_s,      label="Gaskessel",               color="firebrick",  linewidth=1.5, linestyle="--")
ax.plot(disc_s,    label="Speicher Entladung",      color="darkorange", linewidth=1.2, linestyle=":")
ax.set_xlabel("Stunden (sortiert nach Wärmebedarf)")
ax.set_ylabel("Leistung [MW_th]")
ax.set_title("Dauerlinie: Wärmeerzeugung WP, Gaskessel und Speicher")
ax.legend(); ax.grid(alpha=0.4)
plt.tight_layout(); plt.show()

# ── Plot 5: PV-Investitions-Cashflow ─────────────────────────────────────────
years        = np.arange(project_lifetime + 1)
cum_cashflow = np.array([sum(cashflows_pv[:y+1]) for y in range(len(cashflows_pv))])
cum_npv_arr  = np.array([npv_func(discount_rate, cashflows_pv[:y+1])
                          for y in range(len(cashflows_pv))])

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(years, cum_cashflow / 1e6, label="Kumulierter Cashflow (nominal)", alpha=0.6)
ax.plot(years, cum_npv_arr / 1e6, color="crimson", linewidth=2,
        label="Kumulierter NPV (diskontiert)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Jahr")
ax.set_ylabel("Mio. €")
ax.set_title(f"PV-Investitionsrechnung (CAPEX = {capex_pv/1e6:.1f} Mio. €, "
             f"Amortisation ≈ {payback_pv:.1f} a)")
ax.legend(); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); plt.show()
