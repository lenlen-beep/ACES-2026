import pandas as pd
from demandlib import bdew

# Import Meteostat library and dependencies
from datetime import datetime
import meteostat as ms
from pathlib import Path

def create_mean_german_building_loads(rated_load):
    # dena Gebäudereport 2024 (Wohngebäudebestand) 
    # verwendete Aufteileung: 66% EFH, 16% ZFH, 15% MFH

    load_EFH = rated_load * 0.66 + rated_load * 0.16
    # load_ZFH = rated_load * 0.16 ZFH gibt es in BDEW nicht
    load_MFH = rated_load * 0.15

    return load_EFH, load_MFH


def load_temperature_data(
    lat,
    lon,
    year,
    reload_data=False,
    cache_dir="src/ACES-2026/Data/weather_cache"
):

    # --------------------------------------------------
    # Cache-Ordner
    # --------------------------------------------------

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Dateiname
    # --------------------------------------------------

    filename = f"weather_{lat:.3f}_{lon:.3f}_{year}.csv"

    filepath = cache_path / filename

    # --------------------------------------------------
    # Cache prüfen
    # --------------------------------------------------

    if filepath.exists() and not reload_data:

        print("Lade Wetterdaten aus Cache ...")

        df = pd.read_csv(
            filepath,
            index_col=0,
            parse_dates=True
        )
        station_id = filepath.stem

    else:

        print("Lade Wetterdaten von Meteostat ...")

        station_id = ms.Stations().nearby(lat, lon, 50000).fetch().index[0]
        print(f"\nVerwendete Station: {station_id}")

        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59)

        df = ms.Hourly(station_id, start, end).fetch()

        # --------------------------------------------------
        # Lokal speichern
        # --------------------------------------------------

        df.to_csv(filepath)

        print(f"\nGespeichert unter:\n{filepath}")

    # --------------------------------------------------
    # Temperatur extrahieren
    # --------------------------------------------------

    temperature = df["temp"]
    
    #  --------------------------------------------------
    # Zeitindex
    # ---------------------------------------------------

    index = pd.date_range(
    start="2024-01-01 00:00",
    end="2024-12-31 23:00",
    freq="1h"
    )

    # Sicherheitscheck:
    temperature = temperature.reindex(index)

    # Fehlende Werte interpolieren
    temperature = temperature.interpolate()


    return temperature, df, index, station_id


def create_bdew_prifles(load_EFH, load_MFH, temperature, time_index):
    # --------------------------------------------------
    # Gebäude definieren
    # --------------------------------------------------

    buildings = [
        {
            "name": "EFH",
            "shlp_type": "EFH",
            "annual_heat_demand": load_EFH 
        },
        {
            "name": "MFH",
            "shlp_type": "MFH",
            "annual_heat_demand": load_MFH
        }
    ]


    # --------------------------------------------------
    # Lastprofile erzeugen
    # --------------------------------------------------

    profiles = {}

    for b in buildings:

        is_residential = b["shlp_type"] in ("EFH", "MFH")
        model = bdew.HeatBuilding(
            time_index,
            temperature=temperature,
            annual_heat_demand=b["annual_heat_demand"],
            shlp_type=b["shlp_type"],
            wind_class=0,
            building_class=1 if is_residential else 0
        )

        profiles[b["name"]] = model.get_bdew_profile()

    # --------------------------------------------------
    # Gesamtlast
    # --------------------------------------------------

    total_load = sum(profiles.values())
    heat_supply = total_load.sum()

    return profiles, total_load, heat_supply




