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
# Read parameters (YAML)
#-------------------------------------------------------------------------

def read_parameters(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


#-------------------------------------------------------------------------
# Read electricity prices (wholesale day-ahead prices 2024)
#-------------------------------------------------------------------------

def read_price_data(path, filename, load_data):
    price_file = path + filename
    df_price = pd.read_excel(price_file, skiprows=9, header=0, usecols=['Datum von', 'Deutschland/Luxemburg [€/MWh]'])

    df_price.columns = ['Datum von', 'Deutschland/Luxemburg [€/MWh]']

    # Rename column
    df_price.rename(columns={'Datum von': 'Datum'}, inplace=True)

    df_price['Datum'] = pd.to_datetime(
        df_price['Datum'],
        format='%d.%m.%Y %H:%M'
    )

    # Rename column
    df_price.rename(columns={'Datum von': 'Datum'}, inplace=True)

    # Align to load_data time index; interpolate missing prices
    df_price = df_price.set_index('Datum')
    df_price = df_price.groupby(level=0).mean()  # merge duplicate timestamps (DST changeover)
    df_price = df_price.reindex(load_data.index)
    df_price = df_price.interpolate(method='time')
    df_price = df_price.reset_index().rename(columns={'index': 'Datum'})

    price = df_price['Deutschland/Luxemburg [€/MWh]'].values 

    return price


#-------------------------------------------------------------------------
# Read gas prices (PEGAS THE DA settlement 2024, daily → hourly)
#-------------------------------------------------------------------------

def read_gas_price_data(path, filename, load_data):
    df = pd.read_excel(path + filename, header=3)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'].dt.year == 2024][['Date', 'Settlement']].set_index('Date')

    # Upsample daily → hourly and align to load_data index
    hourly_index = pd.date_range(start='2024-01-01', end='2024-12-31 23:00', freq='1h')
    df = df.reindex(hourly_index).interpolate(method='time')
    df = df.reindex(load_data.index).interpolate(method='time')

    return df['Settlement'].values


#-------------------------------------------------------------------------
# Load temperature data
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





