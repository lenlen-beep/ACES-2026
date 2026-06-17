import pandas as pd

load_file_2014_2016  = r"src/Testprojects/Data/district-heating-network-data-flensburg-2014-2016.csv"
load_file_2017_2019  = r"src/Testprojects/Data/2017-2019 Stadtwerke Flensburg Heat Network Data Hourly.xlsx"
load_file_2020_2024  = r"src/Testprojects/Data/2020-2024 Stadtwerke Flensburg Heat Network Data Hourly.xlsx"


# pd.readexcel(file,
#               skiprows: Zeilen überspringen (Anzahl),
#               header: Zeile mit Überschrift,
#               usecols:=[] Bestimmte Spaltenauswahl,
#               names=[]: Spalten direkt benennen,
#               parse_dates=[]: direkt in Datum umwandeln,
#               index_col: Index setzen,
#               na_values=['','']: Behandlung fehlender Werte,
#               nrows: Wertebereich laden (mit skiprows),
#   )

df_2014_2016 = pd.read_csv(load_file_2014_2016, header=0, usecols=['Datetime', 'Overall heat load in MW'])
df_2014_2016.rename(columns={'Datetime': 'Datum', 'Overall heat load in MW': 'Wärmeleistung in MW'}, inplace=True)
df_2014_2016['Datum'] = pd.to_datetime(df_2014_2016['Datum'], format='%d/%m/%y %H:%M')

df_2017_2019 = pd.read_excel(load_file_2017_2019, skiprows=1, header=0, usecols=[0, 1])
df_2017_2019.columns = ['Datum', 'Wärmeleistung in MW']
df_2017_2019['Datum'] = pd.to_datetime(df_2017_2019['Datum'])

df_2020_2024 = pd.read_excel(load_file_2020_2024, skiprows=1, header=0, usecols=[0, 1])
df_2020_2024.columns = ['Datum', 'Wärmeleistung in MW']
df_2020_2024['Datum'] = pd.to_datetime(df_2020_2024['Datum'])

df_all = pd.concat([df_2014_2016, df_2017_2019, df_2020_2024], ignore_index=True)
df_all['Wärmeleistung in MW'] = pd.to_numeric(df_all['Wärmeleistung in MW'], errors='coerce')

df_all['Jahr'] = df_all['Datum'].dt.year
df_all['Monat_Tag_Stunde'] = df_all['Datum'].dt.strftime('%m-%d %H:%M')

df_years = df_all.pivot(index='Monat_Tag_Stunde', columns='Jahr', values='Wärmeleistung in MW')

import matplotlib.pyplot as plt

# Unsortiert
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

for year in df_years.columns:
    ax1.plot(df_years[year].values, label=str(year), alpha=0.7, linewidth=0.8)
ax1.set_title("Wärmeleistung – Zeitreihe")
ax1.set_xlabel("Stunde im Jahr")
ax1.set_ylabel("Wärmeleistung [MW]")
ax1.legend()
ax1.grid(True)

# Sortiert (Dauerlinie)
for year in df_years.columns:
    ax2.plot(sorted(df_years[year].dropna(), reverse=True), label=str(year), alpha=0.7, linewidth=0.8)
ax2.set_title("Wärmeleistung – Dauerlinie (sortiert)")
ax2.set_xlabel("Stunden (sortiert)")
ax2.set_ylabel("Wärmeleistung [MW]")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

from datetime import datetime
from pathlib import Path
import meteostat as ms

LAT, LON = 54.78, 9.43
CACHE_DIR = "src/Testprojects/Data/weather_cache"

def load_temperature_year(year, lat=LAT, lon=LON, cache_dir=CACHE_DIR):
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    filepath = cache_path / f"weather_{lat:.3f}_{lon:.3f}_{year}.csv"

    if filepath.exists():
        print(f"  {year}: Cache")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        print(f"  {year}: Meteostat")
        nearby = ms.stations.nearby(ms.Point(lat, lon), limit=1)
        station_id = nearby.index[0]
        ts = ms.hourly(station_id, datetime(year, 1, 1), datetime(year, 12, 31, 23, 59))
        df = ts.fetch()
        df.to_csv(filepath)

    return df["temp"]


# Temperaturen für alle Jahre laden
print("Lade Temperaturdaten 2014–2024 ...")
temp_series = {}
for year in range(2014, 2025):
    temp_series[year] = load_temperature_year(year)

# Temperaturen an df_all anfügen
df_all = df_all.set_index('Datum')
df_all['Temperatur °C'] = pd.concat(temp_series.values())
df_all = df_all.dropna(subset=['Wärmeleistung in MW', 'Temperatur °C'])

# Plot: Außentemperatur (x) vs. Wärmeleistung (y), ein Scatter pro Jahr
fig, ax = plt.subplots(figsize=(10, 6))
for year in range(2014, 2025):
    mask = df_all['Jahr'] == year
    ax.scatter(
        df_all.loc[mask, 'Temperatur °C'],
        df_all.loc[mask, 'Wärmeleistung in MW'],
        label=str(year), alpha=0.3, s=2
    )
ax.set_xlabel("Außentemperatur [°C]")
ax.set_ylabel("Wärmeleistung [MW]")
ax.set_title("Außentemperatur vs. Wärmeleistung (2014–2024)")
ax.legend(markerscale=5)
ax.grid(True)
plt.tight_layout()
plt.show()


