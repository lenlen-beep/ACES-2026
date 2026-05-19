import pandas as pd
import yaml

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

