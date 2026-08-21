"""
Sensitivitaetsanalyse fuer das ACES-Energiesystemmodell.

Aufruf aus dem Repo-Wurzelverzeichnis:

    python src/ACES-2026/sensitivity.py              # alle Szenarien
    python src/ACES-2026/sensitivity.py --list       # Szenarien anzeigen
    python src/ACES-2026/sensitivity.py --only base gas_high

Ergebnisse landen in src/ACES-2026/Data/sensitivity_results.csv.

Arbeitsweise
------------
Jedes Szenario laeuft in einem eigenen Python-Prozess. Das ist noetig, weil
funcs.energy_system_optimization und funcs.LCOH die Parameter beim Import in
Modulkonstanten einlesen -- eine Aenderung des parameters-Dicts zur Laufzeit
wuerde dort nicht ankommen. Der Runner schreibt daher die veraenderte
parameters.yaml auf die Platte, startet den Teilprozess und stellt das Original
danach wieder her.

Die Netzsimulation wird uebersprungen: die Lastreihe kommt aus der zuvor
erzeugten Data/result_timeseries.csv. Das ist zulaessig, solange ein Szenario
weder Netzgeometrie noch Temperaturen aendert. Szenarien, die
net_parameters.supply_temperature oder delta_T anfassen, sind deshalb NICHT
enthalten -- die brauchen einen vollen main.py-Lauf.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ACES_DIR = Path(__file__).resolve().parent
REPO_ROOT = ACES_DIR.parent.parent
PARAMS = ACES_DIR / "parameters.yaml"
PARAMS_BACKUP = ACES_DIR / "parameters.yaml.sensitivity_backup"
TIMESERIES = ACES_DIR / "Data" / "result_timeseries.csv"
TRASSE = ACES_DIR / "Data" / "Trassierung_Jerrishoe.gpkg"
OUT_CSV = ACES_DIR / "Data" / "sensitivity_results.csv"

RESULT_MARKER = "###SENSITIVITY_RESULT###"


# ---------------------------------------------------------------------------
# Szenariendefinition
# ---------------------------------------------------------------------------
# Schluessel sind Punktpfade in parameters.yaml.
#
# CO2-Szenarien: der Gastarif ist ein All-in-Preis, der CO2-Anteil steckt darin.
# Basis 45 EUR/t entspricht 0.82 ct/kWh (0.1814 t CO2/MWh). Eine Erhoehung auf
# X EUR/t schlaegt mit (X - 45) * 0.1814 / 10 ct/kWh auf den Tarif durch.
_CO2_FACTOR_T_PER_MWH = 0.1814


def _gas_with_co2(eur_per_t, base_tariff=7.2, base_co2=45.0):
    return round(base_tariff + (eur_per_t - base_co2) * _CO2_FACTOR_T_PER_MWH / 10.0, 3)


# Die Laeufe sind zu Szenarien gruppiert: ein Szenario entspricht einem
# variierten Parameter, seine Varianten sind die einzelnen Laeufe.
#
#   "group"       -> Szenarionummer (Referenzfall: None)
#   "scenario"    -> Name des Szenarios, identisch fuer alle Varianten
#   "variant"     -> Bezeichnung der Variante innerhalb des Szenarios
#   "yaml"        -> Punktpfade in parameters.yaml
#   "elec_factor" -> Faktor auf die Spot-Strompreisreihe (kein yaml-Parameter)
SCENARIOS = {
    "base": {"group": None, "scenario": "Reference case", "variant": "--"},

    # --- Szenario 1: Netzinvestitionskosten ---------------------------------
    "pipe_cost_minus30": {"group": 1, "scenario": "Pipe investment cost",
                          "variant": "700 EUR/m (-30 %)",
                          "yaml": {"pipe_parameters.specific_invest_pipe": 700}},
    "pipe_cost_plus30":  {"group": 1, "scenario": "Pipe investment cost",
                          "variant": "1,300 EUR/m (+30 %)",
                          "yaml": {"pipe_parameters.specific_invest_pipe": 1300}},
    "pipe_cost_carmen":  {"group": 1, "scenario": "Pipe investment cost",
                          "variant": "400 EUR/m (literature)",
                          "yaml": {"pipe_parameters.specific_invest_pipe": 400}},

    # --- Szenario 2: Waermepumpen-Investitionskosten ------------------------
    "hp_cost_minus30": {"group": 2, "scenario": "Heat pump investment cost",
                        "variant": "-30 %",
                        "yaml": {"system_parameters.HP.specific_invest_hp": 840000}},
    "hp_cost_plus30":  {"group": 2, "scenario": "Heat pump investment cost",
                        "variant": "+30 %",
                        "yaml": {"system_parameters.HP.specific_invest_hp": 1560000}},

    # --- Szenario 3: Speicher-Investitionskosten ----------------------------
    # Offset und spezifischer Anteil werden gemeinsam skaliert, sonst waere nur
    # ein Bruchteil der Speicherkosten variiert (Offset dominiert bei kleinen
    # Volumina).
    "storage_cost_minus30": {"group": 3, "scenario": "Storage investment cost",
                             "variant": "-30 %", "yaml": {
        "system_parameters.storage.specific_invest_storage": 155.897,
        "system_parameters.storage.invest_offset_storage": 54614.7}},
    "storage_cost_plus30": {"group": 3, "scenario": "Storage investment cost",
                            "variant": "+30 %", "yaml": {
        "system_parameters.storage.specific_invest_storage": 289.523,
        "system_parameters.storage.invest_offset_storage": 101427.3}},

    # --- Szenario 4: Strommarktpreis ----------------------------------------
    # Nur die Marktkomponente wird skaliert; der Aufschlag aus Netzentgelt,
    # Stromsteuer, Umlagen und Konzessionsabgabe bleibt unveraendert.
    "elec_price_minus30": {"group": 4, "scenario": "Electricity market price",
                           "variant": "-30 %", "elec_factor": 0.7},
    "elec_price_plus30":  {"group": 4, "scenario": "Electricity market price",
                           "variant": "+30 %", "elec_factor": 1.3},

    # --- Szenario 5: Gaspreis -----------------------------------------------
    "gas_low":  {"group": 5, "scenario": "Gas price", "variant": "3.3 ct/kWh",
                 "yaml": {"price_parameters.gas.tarif.usual_mid": 3.3}},
    "gas_high": {"group": 5, "scenario": "Gas price", "variant": "12.2 ct/kWh",
                 "yaml": {"price_parameters.gas.tarif.usual_mid": 12.2}},

    # --- Szenario 6: CO2-Preis (ueber den Gastarif) -------------------------
    "co2_65":  {"group": 6, "scenario": "CO2 price", "variant": "65 EUR/t",
                "yaml": {"price_parameters.gas.tarif.usual_mid": _gas_with_co2(65)}},
    "co2_100": {"group": 6, "scenario": "CO2 price", "variant": "100 EUR/t",
                "yaml": {"price_parameters.gas.tarif.usual_mid": _gas_with_co2(100)}},

    # --- Szenario 7: Kapitalkosten ------------------------------------------
    "interest_3pct": {"group": 7, "scenario": "Discount rate", "variant": "3 %",
                      "yaml": {"invest_parameters.interest_rate": 0.03}},
    "interest_7pct": {"group": 7, "scenario": "Discount rate", "variant": "7 %",
                      "yaml": {"invest_parameters.interest_rate": 0.07}},
}


def scenario_id(name):
    g = SCENARIOS[name].get("group")
    return "Reference" if g is None else f"Scenario {g}"


# ---------------------------------------------------------------------------
# Worker: ein einzelnes Szenario rechnen
# ---------------------------------------------------------------------------

def run_single(scenario_name, elec_factor=1.0):
    """Laeuft im Teilprozess. Gibt das Ergebnis als JSON auf stdout aus."""
    sys.path.insert(0, str(ACES_DIR))

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import geopandas as gpd

    from funcs.paths import PARAMETERS_FILE
    from funcs.read_data import (read_parameters, read_price_data,
                                 read_gas_price_data)
    from funcs.era5_weather import (load_era5_weather, compute_pv_generation,
                                    compute_cop, LAT as ERA5_LAT, LON as ERA5_LON)
    from funcs.energy_system_optimization import optimize_energy_system
    from funcs.LCOH import calculate_lcoh

    parameters = read_parameters(PARAMETERS_FILE)

    # --- Lastreihe aus der zwischengespeicherten Netzsimulation --------------
    df = pd.read_csv(TIMESERIES)
    cp = parameters["net_parameters"]["cp"]
    dt_ist = df["t_supply_k"] - df["t_return_k"]
    df["load_MW"] = df["mdot_kg_per_s"] * cp * dt_ist / 1e6
    load = df.set_index("Datum")["load_MW"]
    load_index_dt = pd.to_datetime(load.index)

    network_length = float(gpd.read_file(TRASSE).to_crs(25832).length.sum())

    # --- Preisreihen --------------------------------------------------------
    ref_2024 = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=8784, freq="1h"))
    data_dir = str(ACES_DIR / "Data") + "/"

    electricity_price = read_price_data(
        path=data_dir,
        filename="Gro_handelspreise_202401010000_202501010000_Stunde.xlsx",
        load_data=ref_2024,
    )
    gas_price = read_gas_price_data(
        path=data_dir,
        filename="Historic_THE_DA_Pegas.xlsx",
        load_data=ref_2024,
    )

    # Schalttag entfernen + 24 h rotieren (identisch zu main.py)
    feb29 = ~((ref_2024.index.month == 2) & (ref_2024.index.day == 29))
    electricity_price = electricity_price[feb29]
    gas_price = gas_price[feb29]
    electricity_price = np.concatenate([electricity_price[24:], electricity_price[:24]])
    gas_price = np.concatenate([gas_price[24:], gas_price[:24]])

    # Szenario-Skalierung der Marktkomponente (vor dem Aufschlag, der in
    # optimize_energy_system bzw. calculate_lcoh addiert wird)
    if elec_factor != 1.0:
        electricity_price = electricity_price * elec_factor

    # --- PV und COP ---------------------------------------------------------
    weather = load_era5_weather(2019, lat=ERA5_LAT, lon=ERA5_LON)
    pv_series = compute_pv_generation(
        weather, lat=ERA5_LAT, lon=ERA5_LON,
        surface_tilt=parameters["system_parameters"]["PV"]["surface_tilt"],
        surface_azimuth=parameters["system_parameters"]["PV"]["surface_azimuth"],
        pv_capacity_MW=1.0,
    )
    pv = pv_series.reindex(load_index_dt).interpolate(method="time").bfill().ffill().values
    cop = compute_cop(weather["T_amb_C"]).reindex(load_index_dt) \
              .interpolate(method="time").bfill().ffill().values

    # --- Optimierung --------------------------------------------------------
    (results, Q_hp, Q_gas, charge, discharge, SOC,
     storage_cap, gas_cap, hp_cap, pv_avail, pv_feed_in, pv_cap,
     seas_charge, seas_discharge, seas_soc, seas_cap) = optimize_energy_system(
        load, electricity_price, gas_price, pv,
        cop=cop,
        elec_price_mode="spot",
        elec_hedge_share=0.0,
        gas_price_mode="tariff",
    )

    lcoh, components = calculate_lcoh(
        demand=load, electricity_price=electricity_price, gas_price=gas_price,
        Q_hp=Q_hp, charge=charge, discharge=discharge, Q_gas_boiler=Q_gas,
        pv_availability=pv_avail, pv_feed_in=pv_feed_in,
        storage_capacity_m3=storage_cap, gas_boiler_capacity=gas_cap,
        pv_capacity=pv_cap, seasonal_capacity_m3=seas_cap,
        network_length=network_length,
        hp_capacity=hp_cap, cop=cop,
        elec_price_mode="spot", gas_price_mode="tariff",
    )

    # --- Kennzahlen ---------------------------------------------------------
    P_grid = np.clip(np.asarray(Q_hp) / cop + np.asarray(pv_feed_in)
                     - np.asarray(pv_avail), 0.0, None)
    vbh = float(P_grid.sum() / P_grid.max()) if P_grid.max() > 0 else 0.0
    vbh_class = "higher_2500VBH" if vbh >= 2500 else "lower_2500VBH"
    assumed = parameters["price_parameters"]["electricity"].get("vbh_class", "lower_2500VBH")

    demand_total = float(np.asarray(load).sum())
    el_hp = float((np.asarray(Q_hp) / cop).sum())

    out = {
        "scenario":          scenario_name,
        "scenario_id":       scenario_id(scenario_name),
        "scenario_name":     SCENARIOS[scenario_name]["scenario"],
        "variant":           SCENARIOS[scenario_name]["variant"],
        "elec_factor":       elec_factor,
        "lcoh_eur_per_mwh":  round(float(lcoh), 2),
        "hp_capacity_mw":    round(float(hp_cap), 3),
        "gas_capacity_mw":   round(float(gas_cap), 3),
        "pv_capacity_mw":    round(float(pv_cap), 3),
        "storage_m3":        round(float(storage_cap), 1),
        "seasonal_m3":       round(float(seas_cap), 1),
        "gas_share_pct":     round(float(np.asarray(Q_gas).sum()) / demand_total * 100, 1),
        "cop_effective":     round(float(np.asarray(Q_hp).sum()) / el_hp, 2) if el_hp > 0 else None,
        "grid_energy_mwh":   round(float(P_grid.sum()), 1),
        "grid_peak_mw":      round(float(P_grid.max()), 3),
        "vbh_h":             round(vbh),
        "vbh_class_actual":  vbh_class,
        "vbh_class_assumed": assumed,
        "vbh_consistent":    vbh_class == assumed,
    }
    for label, vals in components.items():
        out[f"comp_{label}"] = round(float(vals["eur_per_year"]), 0)

    print(RESULT_MARKER + json.dumps(out))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _set_nested(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"Parameter '{dotted}' existiert nicht in parameters.yaml")
    d[keys[-1]] = value


def run_all(names):
    import yaml
    import pandas as pd

    if not TIMESERIES.exists():
        sys.exit(f"FEHLER: {TIMESERIES} fehlt.\n"
                 "Erst einmal main.py vollstaendig laufen lassen, damit die "
                 "Netzsimulation zwischengespeichert wird.")

    original = PARAMS.read_text(encoding="utf-8")
    shutil.copy(PARAMS, PARAMS_BACKUP)

    rows = []
    try:
        for i, name in enumerate(names, 1):
            spec = SCENARIOS[name]
            overrides = spec.get("yaml", {})
            elec_factor = spec.get("elec_factor", 1.0)
            desc = dict(overrides)
            if elec_factor != 1.0:
                desc["elec_price_factor"] = elec_factor
            spec_id = scenario_id(name)
            print(f"\n[{i}/{len(names)}] {spec_id}: {spec['scenario']} "
                  f"[{spec['variant']}]", flush=True)

            cfg = yaml.safe_load(original)
            for path, value in overrides.items():
                _set_nested(cfg, path, value)
            PARAMS.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", name,
                 "--elec-factor", str(elec_factor)],
                cwd=str(REPO_ROOT), capture_output=True, text=True,
            )

            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith(RESULT_MARKER)), None)
            if line is None:
                print("  FEHLGESCHLAGEN. Letzte Zeilen des Teilprozesses:")
                print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
                continue

            res = json.loads(line[len(RESULT_MARKER):])
            res["overrides"] = json.dumps(desc)
            rows.append(res)
            print(f"  LCOH {res['lcoh_eur_per_mwh']:.1f} EUR/MWh | "
                  f"WP {res['hp_capacity_mw']:.3f} MW | "
                  f"VBH {res['vbh_h']} h "
                  f"({'ok' if res['vbh_consistent'] else 'STUFE PRUEFEN!'})")
    finally:
        # Original immer zurueckschreiben, auch bei Abbruch mit Strg+C
        PARAMS.write_text(original, encoding="utf-8")
        PARAMS_BACKUP.unlink(missing_ok=True)
        print("\nparameters.yaml wiederhergestellt.")

    if rows:
        df = pd.DataFrame(rows).set_index("scenario")
        df.to_csv(OUT_CSV)
        print(f"\nErgebnisse: {OUT_CSV}\n")
        cols = ["scenario_id", "scenario_name", "variant", "lcoh_eur_per_mwh",
                "hp_capacity_mw", "pv_capacity_mw", "gas_share_pct",
                "vbh_h", "vbh_consistent"]
        print(df[cols].to_string())

        if "base" in df.index:
            base = df.loc["base", "lcoh_eur_per_mwh"]
            print("\nAbweichung vom Referenzfall:")
            for name, row in df.iterrows():
                if name == "base":
                    continue
                d = row["lcoh_eur_per_mwh"] - base
                print(f"  {row['scenario_id']:12s} {row['scenario_name']:26s} "
                      f"{row['variant']:22s} {d:+8.1f} EUR/MWh "
                      f"({d / base * 100:+5.1f} %)")

        bad = df[~df["vbh_consistent"]]
        if len(bad):
            print("\nWARNUNG: Netzentgeltstufe passt in diesen Szenarien nicht "
                  "zur Annahme in parameters.yaml.")
            print("Dort vbh_class umstellen und die betroffenen Szenarien "
                  "einzeln wiederholen:")
            for name in bad.index:
                print(f"  {bad.loc[name, 'scenario_id']} ({name}): "
                      f"{bad.loc[name, 'vbh_class_actual']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", metavar="NAME", help=argparse.SUPPRESS)
    ap.add_argument("--elec-factor", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="nur diese Szenarien rechnen")
    ap.add_argument("--list", action="store_true", help="Szenarien anzeigen")
    args = ap.parse_args()

    if args.list:
        seen = set()
        for name, spec in SCENARIOS.items():
            sid = scenario_id(name)
            head = sid if sid not in seen else ""
            title = spec["scenario"] if sid not in seen else ""
            seen.add(sid)
            print(f"  {head:12s} {title:26s} {spec['variant']:22s} ({name})")
        return

    if args.worker:
        run_single(args.worker, args.elec_factor)
        return

    names = args.only or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"Unbekannte Szenarien: {', '.join(unknown)}")
    run_all(names)


if __name__ == "__main__":
    main()
