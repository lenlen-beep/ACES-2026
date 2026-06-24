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
# Einlesen Wärmedaten (Flensburg 2014–2024, stündlich)
#-------------------------------------------------------------------------
def read_heat_data_FL(scale_factor=1.0):
    load_file_2014_2016 = r"src/ACES2026/Data/district-heating-network-data-flensburg-2014-2016.csv"
    load_file_2017_2019 = r"src/ACES2026/Data/2017-2019 Stadtwerke Flensburg Heat Network Data Hourly.xlsx"
    load_file_2020_2024 = r"src/ACES2026/Data/2020-2024 Stadtwerke Flensburg Heat Network Data Hourly.xlsx"

    df_2014_2016 = pd.read_csv(load_file_2014_2016, header=0,
                                usecols=['Datetime', 'Overall heat load in MW'])
    df_2014_2016.rename(columns={'Datetime': 'Datum',
                                'Overall heat load in MW': 'Wärmeleistung in MW'}, inplace=True)
    df_2014_2016['Datum'] = pd.to_datetime(df_2014_2016['Datum'], format='%d/%m/%y %H:%M')
    df_2017_2019 = pd.read_excel(load_file_2017_2019, skiprows=1, header=0, usecols=[0, 1])
    df_2017_2019.columns = ['Datum', 'Wärmeleistung in MW']
    df_2017_2019['Datum'] = pd.to_datetime(df_2017_2019['Datum'])
    df_2020_2024 = pd.read_excel(load_file_2020_2024, skiprows=1, header=0, usecols=[0, 1])
    df_2020_2024.columns = ['Datum', 'Wärmeleistung in MW']
    df_2020_2024['Datum'] = pd.to_datetime(df_2020_2024['Datum'])

    df_load = pd.concat([df_2014_2016, df_2017_2019, df_2020_2024], ignore_index=True)
    df_load['Wärmeleistung in MW'] = pd.to_numeric(df_load['Wärmeleistung in MW'], errors='coerce')
    df_load = df_load.set_index('Datum').sort_index()
    df_load['Wärmeleistung in MW'] *= scale_factor
    return df_load

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

    return df["temp"]


#-------------------------------------------------------------------------
# Wärme- und Temperaturdaten zusammenführen
#-------------------------------------------------------------------------

def join_heat_and_temp_data(df_load, df_T):
    df = df_load.join(df_T.rename('Temperatur °C'), how='inner').dropna()
    df = df[df['Wärmeleistung in MW'] > 0]
    df['Jahr'] = df.index.year

    T_arr = df['Temperatur °C'].values
    Q_arr = df['Wärmeleistung in MW'].values
    N_m   = len(T_arr)

    # Absolute Stunden-Indizes für den gemergten Ausschnitt (relativ zu 2014-01-01)
    x_abs = ((df.index - pd.Timestamp('2014-01-01')).total_seconds() / 3600).values
    print(f"Gemeinsame Datenpunkte: {N_m}  ({N_m/8760:.1f} Jahre)")

    return df, T_arr, Q_arr, x_abs




