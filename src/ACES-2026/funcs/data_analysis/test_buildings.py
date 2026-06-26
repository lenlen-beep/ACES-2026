import pandas as pd

DATA_DIR = "src/ACES-2026/Data/Aalborg_smart_meter_data/"

# Beispielgebäude aus 1.csv mit nahezu vollständigen 2019-Daten
EXAMPLE_FILE = DATA_DIR + "1.csv"
EXAMPLE_IDS  = [17, 25, 38]


def load_example_buildings(filepath: str = EXAMPLE_FILE,
                           customer_ids: list = EXAMPLE_IDS) -> pd.DataFrame:
    """
    Lädt drei Beispielgebäude aus den Aalborg Smart Meter CSVs.
    Rückgabe: DataFrame mit Spalte 'Datum' (stündlich, 2019) und
              Spalten '1', '2', '3' (Effekt 1 in kW).
    """
    df_raw = pd.read_csv(filepath,
                         usecols=["CustomerID", "RoundedReadTime", "Effekt 1"],
                         low_memory=False)
    df_raw["RoundedReadTime"] = pd.to_datetime(df_raw["RoundedReadTime"],
                                               dayfirst=True, errors="coerce")
    df_raw["Effekt 1"] = pd.to_numeric(df_raw["Effekt 1"], errors="coerce")
    df_raw = df_raw[df_raw["RoundedReadTime"].dt.year == 2019]

    index_2019 = pd.date_range("2019-01-01", periods=8760, freq="1h")
    df_out = pd.DataFrame({"Datum": index_2019})

    for col_num, cid in enumerate(customer_ids, start=1):
        agg = (df_raw[df_raw["CustomerID"] == cid]
               .groupby("RoundedReadTime")["Effekt 1"]
               .mean()
               .reset_index())
        agg.columns = ["Datum", str(col_num)]
        df_out = df_out.merge(agg, on="Datum", how="left")

    # Lücken füllen
    for col in ["1", "2", "3"]:
        df_out[col] = df_out[col].interpolate(method="index").bfill().ffill()

    return df_out



df = load_example_buildings()
print(df.head(10))
print(f"\nForm: {df.shape}  |  NaN gesamt: {df.isna().sum().sum()}")
