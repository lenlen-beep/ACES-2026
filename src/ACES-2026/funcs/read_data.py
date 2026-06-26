import pandas as pd
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
import meteostat as ms

# default
LAT       = 54.78
LON       = 9.43
CACHE_DIR = "src/ACES-2026/Data/weather_cache"

#-------------------------------------------------------------------------
# Einlesen Parameter (YAML)
#-------------------------------------------------------------------------

def read_parameters(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


#-------------------------------------------------------------------------
# Einlesen Strompreise (Großhandelsstrompreise 2024, dummy)
#-------------------------------------------------------------------------

def read_price_data(path, filename, load_data):
    price_file = path + filename
    df_price = pd.read_excel(price_file, skiprows=9, header=0, usecols=['Datum von', 'Deutschland/Luxemburg [€/MWh]'])

    df_price.columns = ['Datum von', 'Deutschland/Luxemburg [€/MWh]']

    # Spalte umbenennen
    df_price.rename(columns={'Datum von': 'Datum'}, inplace=True)

    df_price['Datum'] = pd.to_datetime(
        df_price['Datum'],
        format='%d.%m.%Y %H:%M'
    )

    # Spalte umbenennen
    df_price.rename(columns={'Datum von': 'Datum'}, inplace=True)

    # Auf den Zeitindex von load_data ausrichten; fehlende Preise interpolieren
    df_price = df_price.set_index('Datum')
    df_price = df_price.groupby(level=0).mean()  # doppelte Zeitstempel (Zeitumstellung) zusammenfassen
    df_price = df_price.reindex(load_data.index)
    df_price = df_price.interpolate(method='time')
    df_price = df_price.reset_index().rename(columns={'index': 'Datum'})

    price = df_price['Deutschland/Luxemburg [€/MWh]'].values 

    return price


#-------------------------------------------------------------------------
# Einlesen Gaspreise (PEGAS THE DA Settlement 2024, täglich → stündlich)
#-------------------------------------------------------------------------

def read_gas_price_data(path, filename, load_data):
    df = pd.read_excel(path + filename, header=3)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2024][['Date', 'Settlement']].set_index('Date')

    # Täglich → stündlich hochskalieren und auf load_data-Index ausrichten
    hourly_index = pd.date_range(start='2024-01-01', end='2024-12-31 23:00', freq='1h')
    df = df.reindex(hourly_index).interpolate(method='time')
    df = df.reindex(load_data.index).interpolate(method='time')

    return df['Settlement'].values


#-------------------------------------------------------------------------
# Einlesen der Solardaten (renewables.ninja, stündlich)
#-------------------------------------------------------------------------

def read_pv_data(path, filename, load_data):
    df = pd.read_csv(path + filename, header=3)

    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "electricity"]]

    # Jahr von 2019 auf 2024 setzen
    df["time"] = df["time"].apply(lambda x: x.replace(year=2024))
    df = df.set_index("time")
    df = df.reindex(load_data.index)

    # Fehlende Werte interpolieren
    df["electricity"] = df["electricity"].interpolate(method="time")

    return df["electricity"].values


#-------------------------------------------------------------------------
# Temperaturdaten laden
#-------------------------------------------------------------------------

def load_temperature_data(year, lat=LAT, lon=LON, cache_dir=CACHE_DIR):
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    filepath = cache_path / f"weather_{lat:.3f}_{lon:.3f}_{year}.csv"
    if filepath.exists():
        print(f"  {year}: Cache")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        print(f"  {year}: Meteostat")
        station_id = ms.Stations().nearby(lat, lon, 50000).fetch().index[0]
        df         = ms.Hourly(station_id,
                               datetime(year, 1, 1),
                               datetime(year, 12, 31, 23, 59)).fetch()
        df.to_csv(filepath)

    index = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:00", freq="1h")
    temp  = df["temp"].reindex(index).interpolate(method="time").bfill().ffill()
    return temp





