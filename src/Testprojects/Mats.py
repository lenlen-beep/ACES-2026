#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:46:43 2026

@author: matsbeyer
"""

import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

"""
Optimierung eines Fernwärmesystems mit Wärmepumpe, Wärmespeicher und Photovoltaik-Anlage.

Das Modell minimiert die Netto-Stromkosten der Wärmepumpe.
PV-Strom reduziert den Netzbezug (Eigenverbrauch) oder wird gegen Vergütung eingespeist.
Reale stündliche Börsenpreise (SMARD 2024) werden verwendet.
Post-Optimierung: wirtschaftliche Kennzahlen inkl. CAPEX, OPEX, IRR.

Methodik:
- Daten: Zeitreihe der Wärmeleistung (Excel), PV-Ertrag (CSV), Börsenpreise (Excel)
- Optimierung: Pyomo (lineares Optimierungsmodell)
- Ziel: Minimierung der Netto-Stromkosten (Netzbezug − Einspeisevergütung)
- Nebenbedingungen: Wärmebilanz, Speicherdynamik, Strombilanz
"""

# =============================================================================
# Dateipfade
# =============================================================================
load_file  = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/district_heating_data_Flensburg_2017.xlsx"
pv_file    = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/ninja_pv_54.7833_9.4333_uncorrected.csv"
price_file = r"/Users/matsbeyer/Documents/GitHub/ACES-2026/src/Testprojects/Gro_handelspreise_202401010000_202501010000_Stunde.xlsx"

# =============================================================================
# Daten laden
# =============================================================================

# Wärmelast [MW_th]
df_load = pd.read_excel(load_file, skiprows=1, header=0)
df_load.columns = ['Datum', 'Wärmeleistung in MW']
df_load['Datum'] = pd.to_datetime(df_load['Datum'])

# PV-Ertrag (renewables.ninja, Flensburg 2015)
# Spalte 'electricity': normierter Ertrag [kW / kW_peak] → dimensionslos
df_pv = pd.read_csv(pv_file, skiprows=3, header=0)
df_pv.columns = ['time', 'local_time', 'electricity', 'missing']
df_pv['electricity'] = pd.to_numeric(df_pv['electricity'], errors='coerce').fillna(0)

# Börsenstrompreise DE/LU (SMARD 2024) [€/MWh_el]
# skiprows=9: Metadaten-Header überspringen
df_price = pd.read_excel(price_file, skiprows=9, header=0)
price_col = 'Deutschland/Luxemburg [€/MWh]'
df_price[price_col] = pd.to_numeric(df_price[price_col], errors='coerce').fillna(0)

# Alle Zeitreihen auf gleiche Länge kürzen (kürzeste Datei bestimmt n)
n = min(len(df_load), len(df_pv), len(df_price))
T = range(n)

demand    = df_load['Wärmeleistung in MW'].values[:n]          # [MW_th]
pv_cf     = df_pv['electricity'].values[:n]                    # [-] Kapazitätsfaktor
price_arr = df_price[price_col].values[:n]                     # [€/MWh_el]
price     = {t: price_arr[t] for t in T}

# =============================================================================
# Parameter
# =============================================================================

# Wärmepumpe & Speicher
COP         = 3.5   # Leistungszahl WP [-]
storage_cap = 80   # Wärmespeicher-Kapazität [MWh_th]
charge_max  = 80    # Max. Lade-/Entladeleistung Speicher [MW_th]
eta         = 0.9   # Speicherwirkungsgrad [-]

# Mindestgröße der WP: in jeder Stunde muss gelten:
#   COP * P_wp_max + charge_max >= max(demand)
# → P_wp_max >= (max(demand) - charge_max) / COP
P_wp_min_required = max(0, (max(demand) - charge_max) / COP)
print(f"Spitzenlast Fernwärme:       {max(demand):.1f} MW_th")
print(f"Mindest-WP-Leistung:         {P_wp_min_required:.1f} MW_el  "
      f"(= (Spitzenlast − {charge_max} MW Speicher) / COP {COP})")

P_wp_max = 20   # Max. el. Leistung WP [MW_el] – muss ≥ P_wp_min_required sein!

if P_wp_max < P_wp_min_required:
    print(f"WARNUNG: P_wp_max = {P_wp_max} MW_el ist zu klein! "
          f"Mindestens {P_wp_min_required:.1f} MW_el erforderlich → Modell wird infeasible.")
else:
    print(f"P_wp_max = {P_wp_max} MW_el  ✓  (ausreichend)")

# Solarpark
# pv_avail[t]: verfügbare PV-Leistung [MW_el] = Kapazitätsfaktor * installierte Leistung
P_pv_peak = 10                          # installierte PV-Leistung [MW_el]
pv_avail  = pv_cf * P_pv_peak           # [MW_el]

# Einspeisevergütung für Utility-Scale-PV (50 MW → kein EEG, Direktvermarktung)
# Muss deutlich unter dem durchschnittlichen Strompreis liegen, damit Eigenverbrauch
# (vermiedener Netzbezug zum Spotpreis) attraktiver ist als Einspeisung.
# Faustregel: ~70-80% des Ø-Spotpreises als konservativer Abnahmepreis.
feed_in_tariff = 50   # [€/MWh_el] – Direktvermarktung mit Abschlag

# Gaskessel (Backup für Spitzenlast – detailliertes Modell folgt später)
eta_gb    = 0.92   # Wirkungsgrad Gaskessel [-]
gas_price = 40     # Gaspreis [€/MWh_th Brennstoff]
# Kein P_gb_max gesetzt → Gaskessel übernimmt unbegrenzt, was WP + Speicher nicht schafft

# Wirtschaftliche Parameter (für CAPEX/OPEX/IRR-Berechnung nach Optimierung)
specific_capex  = 700_000   # Investitionskosten Solarpark [€/MW_el installiert]
opex_rate       = 0.015     # jährl. Betriebskosten als Anteil des CAPEX [-]
project_lifetime = 20       # Projektlaufzeit [Jahre]
discount_rate   = 0.06      # Kalkulationszinssatz [-]

# =============================================================================
# Hilfsfunktionen
# =============================================================================

def npv_func(rate, cashflows):
    return sum(cf / (1 + rate)**t for t, cf in enumerate(cashflows))


def run_optimization(pv_peak):
    """Optimiert das System für eine gegebene PV-Spitzenleistung [MW_el].

    Gibt ein dict mit Zeitreihen und wirtschaftlichen Kennzahlen zurück.
    """
    pv = pv_cf * pv_peak  # verfügbare PV-Leistung [MW_el]

    m   = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=T)

    m.P_wp      = pyo.Var(m.T, bounds=(0, P_wp_max))
    m.charge    = pyo.Var(m.T, bounds=(0, charge_max))
    m.discharge = pyo.Var(m.T, bounds=(0, charge_max))
    m.SOC       = pyo.Var(m.T, bounds=(0, storage_cap))
    m.P_grid    = pyo.Var(m.T, bounds=(0, None))
    m.P_feed    = pyo.Var(m.T, bounds=(0, None))
    m.Q_gb      = pyo.Var(m.T, bounds=(0, None))  # Gaskessel-Wärmeleistung [MW_th]

    def obj_rule(m):
        stromkosten = sum(price[t] * m.P_grid[t] - feed_in_tariff * m.P_feed[t] for t in m.T)
        gaskosten   = sum((gas_price / eta_gb) * m.Q_gb[t] for t in m.T)
        return stromkosten + gaskosten
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def heat_balance(m, t):
        return COP * m.P_wp[t] + m.discharge[t] + m.Q_gb[t] == demand[t] + m.charge[t]
    m.heat_balance = pyo.Constraint(m.T, rule=heat_balance)

    def storage_rule(m, t):
        if t == 0:
            return m.SOC[t] == 0
        return m.SOC[t] == m.SOC[t-1] + eta * m.charge[t] - (1/eta) * m.discharge[t]
    m.storage = pyo.Constraint(m.T, rule=storage_rule)

    def electricity_balance(m, t):
        return m.P_grid[t] + pv[t] == m.P_wp[t] + m.P_feed[t]
    m.elec_balance = pyo.Constraint(m.T, rule=electricity_balance)

    def feed_limit(m, t):
        return m.P_feed[t] <= pv[t]
    m.feed_lim = pyo.Constraint(m.T, rule=feed_limit)

    solver = pyo.SolverFactory('glpk')
    solver.solve(m)

    P_wp_r   = np.array([pyo.value(m.P_wp[t])      for t in T])
    charge_r = np.array([pyo.value(m.charge[t])     for t in T])
    disc_r   = np.array([pyo.value(m.discharge[t])  for t in T])
    P_grid_r = np.array([pyo.value(m.P_grid[t])     for t in T])
    P_feed_r = np.array([pyo.value(m.P_feed[t])     for t in T])
    Q_gb_r   = np.array([pyo.value(m.Q_gb[t])       for t in T])
    P_self_r = pv - P_feed_r

    return {
        'P_wp':   P_wp_r,
        'charge': charge_r,
        'disc':   disc_r,
        'P_grid': P_grid_r,
        'P_feed': P_feed_r,
        'P_self': P_self_r,
        'Q_gb':   Q_gb_r,
        'pv':     pv,
    }


def compute_economics(res, pv_peak):
    """Berechnet wirtschaftliche Kennzahlen aus Optimierungsergebnissen."""
    gas_cost        = (gas_price / eta_gb) * res['Q_gb'].sum()
    cost_with_pv    = price_arr @ res['P_grid'] - feed_in_tariff * res['P_feed'].sum() + gas_cost
    cost_without_pv = price_arr @ res['P_wp'] + gas_cost
    annual_savings  = cost_without_pv - cost_with_pv  # Einsparung nur durch PV

    capex           = pv_peak * specific_capex
    opex_annual     = opex_rate * capex
    net_savings     = annual_savings - opex_annual
    cashflows       = [-capex] + [net_savings] * project_lifetime
    npv_val         = npv_func(discount_rate, cashflows)

    try:
        irr_val = brentq(lambda r: npv_func(r, cashflows), -0.99, 10.0)
    except ValueError:
        irr_val = None

    payback = capex / net_savings if net_savings > 0 else float('inf')

    return {
        'cost_with_pv':    cost_with_pv,
        'cost_without_pv': cost_without_pv,
        'annual_savings':  annual_savings,
        'net_savings':     net_savings,
        'capex':           capex,
        'opex':            opex_annual,
        'npv':             npv_val,
        'irr':             irr_val,
        'payback':         payback,
        'cashflows':       cashflows,
        'feed_in_revenue': feed_in_tariff * res['P_feed'].sum(),
        'pv_self_value':   price_arr @ res['P_self'],
        'total_heat':      (COP * res['P_wp']).sum(),
    }


# =============================================================================
# Hauptoptimierung (gewählte PV-Größe)
# =============================================================================

solver  = pyo.SolverFactory('glpk')  # wird intern in run_optimization genutzt

res_main  = run_optimization(P_pv_peak)
econ_main = compute_economics(res_main, P_pv_peak)

P_wp_res   = res_main['P_wp']
charge_res = res_main['charge']
disc_res   = res_main['disc']
P_grid_res = res_main['P_grid']
P_feed_res = res_main['P_feed']
P_self_res = res_main['P_self']
pv_avail   = res_main['pv']
demand     = np.array(demand)

# =============================================================================
# Wirtschaftliche Kennzahlen (Hauptoptimierung)
# =============================================================================

cost_with_pv      = econ_main['cost_with_pv']
cost_without_pv   = econ_main['cost_without_pv']
annual_savings_op = econ_main['annual_savings']
feed_in_revenue   = econ_main['feed_in_revenue']
pv_self_value     = econ_main['pv_self_value']
total_heat        = econ_main['total_heat']
specific_cost     = cost_with_pv / total_heat
capex             = econ_main['capex']
opex_annual       = econ_main['opex']
net_savings_annual= econ_main['net_savings']
cashflows         = econ_main['cashflows']
npv_value         = econ_main['npv']
irr               = econ_main['irr']
payback           = econ_main['payback']

# =============================================================================
# Ausgabe
# =============================================================================

print("=" * 50)
print("  SYSTEMKENNZAHLEN")
print("=" * 50)
Q_gb_res = res_main['Q_gb']
print(f"  Gaskessel-Einsatz:          {Q_gb_res.sum():>10.0f} MWh_th/a  "
      f"({(Q_gb_res > 0.01).sum()} Betriebsstunden)")
print(f"  Gaskessel-Spitzenlast:      {Q_gb_res.max():>10.1f} MW_th")
print()
print(f"  PV-Gesamtertrag verfügbar:  {pv_avail.sum():>10.0f} MWh_el/a")
print(f"  PV-Eigenverbrauch:          {P_self_res.sum():>10.0f} MWh_el/a")
print(f"  Einspeisung ins Netz:       {P_feed_res.sum():>10.0f} MWh_el/a")
print(f"  Eigenverbrauchsquote:       {P_self_res.sum() / pv_avail.sum() * 100:>9.1f} %")
print(f"  Netzbezug WP gesamt:        {P_grid_res.sum():>10.0f} MWh_el/a")
print(f"  Wärmeerzeugung gesamt:      {total_heat:>10.0f} MWh_th/a")
print()
print("=" * 50)
print("  BETRIEBSWIRTSCHAFT (Jahr 1)")
print("=" * 50)
print(f"  Stromkosten ohne PV:        {cost_without_pv:>10.0f} €/a")
print(f"  Stromkosten mit PV:         {cost_with_pv:>10.0f} €/a")
print(f"    davon Eigenverbrauch:     {pv_self_value:>10.0f} €/a  (vermiedene Kosten)")
print(f"    davon Einspeiseverg.:     {feed_in_revenue:>10.0f} €/a")
print(f"  Operative Einsparung:       {annual_savings_op:>10.0f} €/a")
print(f"  Spez. Wärmekosten:          {specific_cost:>10.2f} €/MWh_th")
print()
print("=" * 50)
print("  INVESTITIONSRECHNUNG")
print("=" * 50)
print(f"  CAPEX ({P_pv_peak:.0f} MW × {specific_capex/1e3:.0f} k€/MW): {capex:>10.0f} €")
print(f"  OPEX ({opex_rate*100:.1f}% p.a.):            {opex_annual:>10.0f} €/a")
print(f"  Netto-Einsparung:           {net_savings_annual:>10.0f} €/a")
print(f"  NPV ({discount_rate*100:.0f}%, {project_lifetime} a):          {npv_value:>10.0f} €")
if irr is not None:
    print(f"  IRR:                        {irr*100:>9.1f} %")
else:
    print(f"  IRR:                         nicht berechenbar")
print(f"  Amortisationszeit:          {payback:>9.1f} Jahre")
print("=" * 50)

# =============================================================================
# Plots
# =============================================================================

# 1) Dauerlinie (sortiert nach Wärmebedarf)
sorted_idx     = np.argsort(-demand)
demand_sorted  = demand[sorted_idx]
wp_sorted      = (COP * P_wp_res)[sorted_idx]
disc_sorted    = disc_res[sorted_idx]
self_sorted    = (COP * P_self_res)[sorted_idx]
gb_sorted      = Q_gb_res[sorted_idx]

plt.figure(figsize=(10, 5))
plt.plot(demand_sorted,  label="Wärmebedarf")
plt.plot(wp_sorted,      label="Wärmepumpe (gesamt)")
plt.plot(disc_sorted,    label="Speicher Entladung")
plt.plot(self_sorted,    label="PV-Eigenverbrauch (Wärmeäquiv.)")
plt.plot(gb_sorted,      label="Gaskessel (Backup)", color="firebrick", linestyle="--")
plt.xlabel("Stunden (sortiert nach Wärmebedarf)")
plt.ylabel("Leistung [MW]")
plt.title("Dauerlinie: WP, Speicher und Solarpark")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 2) Speicher Lade-/Entladevorgänge
plt.figure(figsize=(10, 4))
plt.plot(charge_res, label="Laden")
plt.plot(disc_res,   label="Entladen")
plt.xlabel("Zeit [h]")
plt.ylabel("Leistung [MW_th]")
plt.title("Speicher Lade- und Entladevorgänge")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 3) Jahresüberblick: PV-Strombilanz (ganzes Jahr, Tagesmittelwerte)
# Stundenwerte werden auf Tagesmittel aggregiert – sonst zu viele Datenpunkte
hours_per_day = 24
n_days = n // hours_per_day

# Tagesmittelwerte [MW_el]
pv_avail_daily = pv_avail[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)
P_self_daily   = P_self_res[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)
P_feed_daily   = P_feed_res[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)
P_grid_daily   = P_grid_res[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)
t_days         = np.arange(n_days)

plt.figure(figsize=(14, 5))
plt.plot(t_days, pv_avail_daily, label="PV verfügbar [MW_el]",
         color="orange", linewidth=1.2)
plt.plot(t_days, P_self_daily,   label="PV-Eigenverbrauch [MW_el]",
         color="gold", linewidth=1.2)
plt.plot(t_days, P_feed_daily,   label="Einspeisung [MW_el]",
         color="limegreen", linewidth=1.2)
plt.plot(t_days, P_grid_daily,   label="Netzbezug WP [MW_el]",
         color="steelblue", linewidth=1.2)

# Monatsbeschriftung (Tage 0, 31, 59, ...)
monatsgrenzen = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
monatsnamen   = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                 "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
plt.xticks(monatsgrenzen, monatsnamen)

plt.xlabel("Monat")
plt.ylabel("Ø Tagesleistung [MW_el]")
plt.title("Jahresüberblick: PV-Strombilanz (Tagesmittelwerte)")
plt.legend(loc="upper right")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

# 4) Jahresüberblick: Wärmedeckung (Tagesmittelwerte)
wp_heat_daily   = (COP * P_wp_res)[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)
gb_daily        = Q_gb_res[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)
disc_daily      = disc_res[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)
demand_daily    = demand[:n_days*hours_per_day].reshape(n_days, hours_per_day).mean(axis=1)

plt.figure(figsize=(14, 5))
plt.plot(t_days, demand_daily,  label="Wärmebedarf [MW_th]",
         color="black",     linewidth=1.8)
plt.plot(t_days, wp_heat_daily, label="Wärmepumpe [MW_th]",
         color="steelblue", linewidth=1.4)
plt.plot(t_days, gb_daily,      label="Gaskessel [MW_th]",
         color="firebrick", linewidth=1.4)
plt.plot(t_days, disc_daily,    label="Speicher Entladung [MW_th]",
         color="darkorange", linewidth=1.2, linestyle="--")
plt.xticks(monatsgrenzen, monatsnamen)
plt.xlabel("Monat")
plt.ylabel("Ø Tagesleistung [MW_th]")
plt.title("Jahresüberblick: Wärmedeckung durch WP, Gaskessel und Speicher (Tagesmittelwerte)")
plt.legend(loc="upper right")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

# 5) Jahres-Cashflow: kumulierter NPV (für gewählte PV-Größe)
years = np.arange(0, project_lifetime + 1)
cum_cashflow = np.array([sum(cashflows[:y+1]) for y in range(len(cashflows))])
cum_npv      = np.array([npv_func(discount_rate, cashflows[:y+1]) for y in range(len(cashflows))])

plt.figure(figsize=(9, 4))
plt.bar(years, cum_cashflow / 1e6, label="Kumulierter Cashflow (nominal)", alpha=0.6)
plt.plot(years, cum_npv / 1e6, color="crimson", linewidth=2, label="Kumulierter NPV (diskontiert)")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Jahr")
plt.ylabel("Mio. €")
plt.title(f"Investitionsrechnung Solarpark (CAPEX = {capex/1e6:.1f} Mio. €)")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# =============================================================================
# Optimale PV-Parkgröße: parametrischer Sweep
# =============================================================================
# Das Modell wird für verschiedene PV-Größen gelöst.
# Gesucht: die Größe mit maximalem NPV (= wirtschaftliches Optimum).

pv_sizes = np.arange(10, 310, 10)   # 10 MW bis 300 MW in 10-MW-Schritten

sweep_npv     = []
sweep_irr     = []
sweep_payback = []
sweep_savings = []

print("\nParametrischer Sweep: optimale PV-Parkgröße wird berechnet ...")
for size in pv_sizes:
    r  = run_optimization(size)
    e  = compute_economics(r, size)
    sweep_npv.append(e['npv'])
    sweep_irr.append(e['irr'] * 100 if e['irr'] is not None else np.nan)
    sweep_payback.append(min(e['payback'], project_lifetime))  # cap bei Laufzeit
    sweep_savings.append(e['net_savings'] / 1e6)
    print(f"  {size:>4.0f} MW  →  NPV = {e['npv']/1e6:>7.2f} Mio. €  |"
          f"  IRR = {sweep_irr[-1]:>5.1f} %  |  Payback = {e['payback']:>5.1f} a")

sweep_npv     = np.array(sweep_npv)
sweep_irr     = np.array(sweep_irr)
sweep_payback = np.array(sweep_payback)
sweep_savings = np.array(sweep_savings)

# Optimale Größe
idx_opt   = np.argmax(sweep_npv)
size_opt  = pv_sizes[idx_opt]
npv_opt   = sweep_npv[idx_opt]

print(f"\n→ Optimale PV-Parkgröße: {size_opt:.0f} MW  (NPV = {npv_opt/1e6:.2f} Mio. €)")

# 6) Sweep-Plot: NPV, IRR und Amortisationszeit vs. PV-Größe
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# NPV
axes[0].plot(pv_sizes, sweep_npv / 1e6, color="steelblue", linewidth=2)
axes[0].axvline(size_opt, color="crimson", linestyle="--", linewidth=1.5,
                label=f"Optimum: {size_opt:.0f} MW")
axes[0].axvline(P_pv_peak, color="gray", linestyle=":", linewidth=1.5,
                label=f"Gewählte Größe: {P_pv_peak:.0f} MW")
axes[0].axhline(0, color="black", linewidth=0.7)
axes[0].set_ylabel("NPV [Mio. €]")
axes[0].set_title("Optimale PV-Parkgröße: Wirtschaftlichkeitsanalyse")
axes[0].legend()
axes[0].grid(alpha=0.4)

# IRR
axes[1].plot(pv_sizes, sweep_irr, color="darkorange", linewidth=2)
axes[1].axvline(size_opt, color="crimson", linestyle="--", linewidth=1.5)
axes[1].axhline(discount_rate * 100, color="black", linestyle=":",
                linewidth=1, label=f"Kalkulationszins ({discount_rate*100:.0f}%)")
axes[1].set_ylabel("IRR [%]")
axes[1].legend()
axes[1].grid(alpha=0.4)

# Amortisationszeit
axes[2].plot(pv_sizes, sweep_payback, color="seagreen", linewidth=2)
axes[2].axvline(size_opt, color="crimson", linestyle="--", linewidth=1.5)
axes[2].axhline(project_lifetime, color="black", linestyle=":",
                linewidth=1, label=f"Projektlaufzeit ({project_lifetime} a)")
axes[2].set_ylabel("Amortisationszeit [Jahre]")
axes[2].set_xlabel("Installierte PV-Leistung [MW_el]")
axes[2].legend()
axes[2].grid(alpha=0.4)

plt.tight_layout()
plt.show()
