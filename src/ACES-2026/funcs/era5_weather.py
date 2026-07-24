"""
Eigenständiges Skript/Modul: ERA5-Wetterdaten (Temperatur, Globalstrahlung, direkte
Strahlung, Windgeschwindigkeit) via Copernicus Climate Data Store (CDS), plus eigenes
PV-Ertragsmodell mit pvlib. Ausführen als Skript liefert eine CSV mit den stündlichen
Wetter- und PV-Daten für ein Jahr (Standard: Jerrishoe, 2019):

    python -m funcs.era5_weather --year 2019

Dies ist (noch) nicht in main.py eingebunden — main.py nutzt weiterhin meteostat
(load_temperature_data) und renewables.ninja (read_pv_data) wie bisher. Die Einbindung
dieses Moduls ins Hauptskript ist ein späterer, separater Schritt.

Setup (einmalig):
    1. Account: https://cds.climate.copernicus.eu (kostenlos)
    2. API-Key ins Profil kopieren, Datei ~/.cdsapirc anlegen:
           url: https://cds.climate.copernicus.eu/api
           key: DEIN-PERSONAL-ACCESS-TOKEN
    3. pip install -e .  (installiert cdsapi, xarray, netCDF4, pvlib)

Quelle: Hersbach et al. (2020), ERA5 global reanalysis, QJRMS 146(730), 1999-2049.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pvlib
import yaml
from pathlib import Path

# default: Jerrishoe (Standort des simulierten Nahwärmenetzes)
LAT = 54.65699858112733
LON = 9.37165074067953

# Absolut statt cwd-relativ, damit es unabhängig vom Arbeitsverzeichnis funktioniert
# (Spyder setzt beim Direktstart eines Skripts sonst dessen eigenen Ordner als cwd).
REPO_SRC_DIR = Path(__file__).resolve().parents[1]   # .../src/ACES-2026
CACHE_DIR    = str(REPO_SRC_DIR / "Data" / "era5_cache")


def _project_parameters():
    """
    Liest parameters.yaml (einzige Quelle der Wahrheit für T_supply_C, eta_carnot,
    performance_ratio, ...). Wird bei jedem Aufruf neu gelesen (keine Caching-Logik),
    damit Änderungen an parameters.yaml sofort wirken, ohne den Kernel neu zu starten.
    Gibt {} zurück, falls die Datei fehlt, statt abzubrechen.
    """
    try:
        with open(REPO_SRC_DIR / "parameters.yaml") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

ERA5_VARIABLES = [
    "2m_temperature",
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


#-------------------------------------------------------------------------
# Download (CDS API)
#-------------------------------------------------------------------------

def _era5_area(lat, lon, buffer_deg=0.25):
    """Bounding Box [Nord, West, Süd, Ost] für die CDS-Anfrage um (lat, lon)."""
    return [lat + buffer_deg, lon - buffer_deg, lat - buffer_deg, lon + buffer_deg]


def _load_single_era5_file(path):
    """Öffnet eine heruntergeladene ERA5-Datei; entpackt transparent, falls die CDS ein Zip liefert."""
    import zipfile
    path = Path(path)
    if zipfile.is_zipfile(path):
        extract_dir = path.with_suffix("")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_dir)
        nc_files = sorted(extract_dir.glob("*.nc"))
        if not nc_files:
            raise RuntimeError(f"Kein .nc im ERA5-Zip gefunden: {path}")
        if len(nc_files) == 1:
            return xr.open_dataset(nc_files[0])
        # Die CDS liefert pro Anfrage oft mehrere .nc im Zip, eine je GRIB-"stepType"
        # (z.B. "instant" fuer t2m/u10/v10, "accum" fuer ssrd/fdir) -- alle mit
        # DENSELBEN Zeitstempeln, aber unterschiedlichen Variablen. Das sind keine
        # Zeitabschnitte zum Verketten, sondern per Variable zusammenzuführen
        # (xr.concat würde hier fälschlich Zeilen duplizieren und die jeweils
        # fehlenden Variablen der anderen Datei mit NaN auffüllen).
        datasets = [xr.open_dataset(f) for f in nc_files]
        return xr.merge(datasets, compat="override", join="inner")
    return xr.open_dataset(path)


def fetch_era5_raw(year, lat=LAT, lon=LON, cache_dir=CACHE_DIR, buffer_deg=0.25, max_parallel=6):
    """
    Lädt stündliche ERA5-Reanalyse (Temperatur, Strahlung, Wind) für ein Jahr
    rund um (lat, lon) von der CDS herunter und cached sie als eine zusammengeführte
    NetCDF-Datei. Erfordert einen eingerichteten CDS-Zugang (~/.cdsapirc, siehe Moduldoc).

    Wird monatsweise abgerufen: Ein Jahr auf einmal überschreitet das "cost limit"
    der CDS ("Your request is too large"). Bereits heruntergeladene Monate werden
    einzeln gecacht, ein abgebrochener Download kann also fortgesetzt werden, ohne
    bereits geladene Monate erneut abzurufen. Fehlende Monate werden bis zu
    `max_parallel` gleichzeitig angefragt (Wartezeit ist serverseitige Warteschlange,
    keine lokale Rechenzeit -> Parallelisieren spart reale Wartezeit). Bei Problemen
    (z.B. Limit an gleichzeitigen Anfragen pro Account) `max_parallel=1` setzen für
    rein sequenzielles Verhalten.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    filepath = cache_path / f"era5_{lat:.3f}_{lon:.3f}_{year}.nc"

    if filepath.exists():
        print(f"  ERA5 {year}: Cache ({filepath})")
        return filepath

    try:
        import cdsapi
    except ImportError as e:
        raise ImportError(
            "cdsapi ist nicht installiert. `pip install -e .` im Repo-Root ausführen."
        ) from e

    try:
        cdsapi.Client()  # nur zum frühen Prüfen, dass ~/.cdsapirc lesbar ist
    except Exception as e:
        raise RuntimeError(
            "CDS-API-Zugang nicht gefunden/eingerichtet. Account unter "
            "https://cds.climate.copernicus.eu anlegen und ~/.cdsapirc anlegen "
            "(siehe Docstring von era5_weather.py)."
        ) from e

    import calendar
    from concurrent.futures import ThreadPoolExecutor, as_completed

    month_dir = cache_path / f"era5_{lat:.3f}_{lon:.3f}_{year}_months"
    month_dir.mkdir(exist_ok=True)

    month_files = [month_dir / f"{m:02d}.nc" for m in range(1, 13)]
    missing = [(m, f) for m, f in zip(range(1, 13), month_files) if not f.exists()]
    for m, f in zip(range(1, 13), month_files):
        if f.exists():
            print(f"  ERA5 {year}-{m:02d}: Cache")

    def _download_month(month, month_file):
        # Eigener Client pro Thread (einfacher/robuster als einen Client zu teilen).
        import cdsapi as _cdsapi
        n_days = calendar.monthrange(year, month)[1]
        _cdsapi.Client().retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ERA5_VARIABLES,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, n_days + 1)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": _era5_area(lat, lon, buffer_deg),
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(month_file),
        )
        return month

    if missing:
        # Alle fehlenden Monate gleichzeitig anfragen statt nacheinander zu warten:
        # Die Wartezeit ist die Warteschlange auf dem CDS-Server, nicht lokale
        # Rechenzeit -> parallele Anfragen laufen dort gleichzeitig statt seriell.
        n_parallel = min(len(missing), max_parallel)
        print(f"  ERA5 {year}: {len(missing)} fehlende Monate parallel abrufen "
              f"(bis zu {n_parallel} gleichzeitig)...")
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(_download_month, m, f): m for m, f in missing}
            for future in as_completed(futures):
                month = futures[future]
                future.result()  # Exception hier weiterreichen, falls ein Monat fehlschlug
                print(f"  ERA5 {year}-{month:02d}: fertig")

    print(f"  ERA5 {year}: {len(month_files)} Monate zusammenführen...")
    month_datasets = [_load_single_era5_file(f) for f in month_files]
    time_dim = "valid_time" if "valid_time" in month_datasets[0].dims else "time"
    ds_year = xr.concat(month_datasets, dim=time_dim).sortby(time_dim)
    ds_year.to_netcdf(filepath)
    ds_year.close()
    for ds in month_datasets:
        ds.close()

    return filepath


def _open_era5_dataset(filepath):
    """Öffnet die gecachte (bereits zusammengeführte) ERA5-Jahresdatei."""
    ds = _load_single_era5_file(filepath)

    # "expver"-Dimension tritt nur bei Anfragen der letzten ~3 Monate auf (Mischung
    # aus finaler ERA5 und vorläufiger ERA5T) -> für historische Jahre irrelevant,
    # hier nur als Sicherheitsnetz behandelt.
    if "expver" in ds.dims:
        ds = ds.mean("expver", skipna=True)
    return ds


#-------------------------------------------------------------------------
# Aufbereitung: NetCDF -> stündlicher DataFrame
#-------------------------------------------------------------------------

def load_era5_weather(year, lat=LAT, lon=LON, cache_dir=CACHE_DIR, buffer_deg=0.25, max_parallel=6):
    """
    Liefert einen stündlichen DataFrame (naiver Index, 1.1. 00:00 - 31.12. 23:00,
    gleiche Konvention wie read_data.load_temperature_data) mit:
        T_amb_C       - 2m-Lufttemperatur [°C]
        wind_speed_ms - 10m-Windgeschwindigkeit [m/s] (aus u10/v10)
        GHI_Wm2       - Globalstrahlung horizontal [W/m²]
        BHI_Wm2       - direkte Strahlung horizontal [W/m²] (ERA5 fdir)
        DHI_Wm2       - diffuse Strahlung horizontal [W/m²] (= GHI - BHI)
        DNI_Wm2       - Direktnormalstrahlung [W/m²] (= BHI / cos(Zenit))
    für den ERA5-Gitterpunkt nächst (lat, lon).
    """
    filepath = fetch_era5_raw(year, lat, lon, cache_dir, buffer_deg, max_parallel)
    ds = _open_era5_dataset(filepath)

    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    df = point.to_dataframe().reset_index()

    # Zeitkoordinate heißt je nach CDS-Backend "time" oder "valid_time"
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.set_index(pd.to_datetime(df[time_col], utc=True)).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    out = pd.DataFrame(index=df.index)
    out["T_amb_C"]       = df["t2m"] - 273.15
    out["GHI_Wm2"]       = (df["ssrd"] / 3600.0).clip(lower=0)
    out["BHI_Wm2"]       = (df["fdir"] / 3600.0).clip(lower=0)
    out["DHI_Wm2"]       = (out["GHI_Wm2"] - out["BHI_Wm2"]).clip(lower=0)
    out["wind_speed_ms"] = np.sqrt(df["u10"]**2 + df["v10"]**2)

    # Auf lückenlosen Stundenindex reindizieren (ERA5 sollte lückenlos sein; nur
    # als Sicherheitsnetz, analog load_temperature_data)
    index_utc = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h", tz="UTC")
    out = out.reindex(index_utc).interpolate(method="time").bfill().ffill()

    # Direktnormalstrahlung (DNI) aus horizontaler Direktstrahlung + Sonnenstand.
    # Physikalisch konsistenter als eine statistische GHI->DNI-Dekomposition (z.B.
    # Erbs-Modell), da ERA5 den direkten Anteil bereits liefert.
    solpos = pvlib.solarposition.get_solarposition(out.index, lat, lon)
    cos_zenith = np.cos(np.radians(solpos["apparent_zenith"])).clip(lower=0.01)
    dni = out["BHI_Wm2"] / cos_zenith
    out["DNI_Wm2"] = dni.where(solpos["apparent_zenith"] < 89, 0).clip(lower=0, upper=1200)

    # Zurück auf naiven lokalen Stundenindex des Zieljahres (Konvention wie
    # load_temperature_data: kein DST-Handling, reine Stunde-für-Stunde-Ausrichtung)
    out = out.tz_localize(None)
    out.index = pd.date_range(f"{year}-01-01", periods=len(out), freq="1h")
    return out


#-------------------------------------------------------------------------
# Temperaturabhängiger COP der Wärmepumpe (Carnot-Ansatz)
#-------------------------------------------------------------------------

def compute_cop(T_amb_C, T_supply_C=None, eta_carnot=None, cop_min=1.5, cop_max=7.0):
    """
    Temperaturabhängiger COP (Carnot-Wirkungsgrad-Ansatz), begrenzt auf [cop_min, cop_max].
    T_supply_C/eta_carnot werden, falls nicht explizit übergeben, aus parameters.yaml
    gelesen (net_parameters.supply_temperature bzw. system_parameters.HP.eta_carnot) --
    fällt auf 80.0/0.45 zurück, falls dort (noch) nicht vorhanden.
    """
    if T_supply_C is None or eta_carnot is None:
        params = _project_parameters()
        if T_supply_C is None:
            T_supply_C = params.get("net_parameters", {}).get("supply_temperature", 80.0)
        if eta_carnot is None:
            eta_carnot = params.get("system_parameters", {}).get("HP", {}).get("eta_carnot", 0.45)

    T_vl_K  = T_supply_C + 273.15
    T_amb_K = T_amb_C + 273.15
    cop = eta_carnot * T_vl_K / (T_vl_K - T_amb_K)
    return cop.clip(lower=cop_min, upper=cop_max)


#-------------------------------------------------------------------------
# Eigenes PV-Ertragsmodell (pvlib) aus ERA5-Strahlungsdaten
#-------------------------------------------------------------------------

def compute_pv_generation(weather, lat=LAT, lon=LON, surface_tilt=35, surface_azimuth=180,
                           pv_capacity_MW=5.0, performance_ratio=None):
    """
    Stündliche PV-Erzeugung [MW] aus ERA5-Strahlungsdaten (weather: DataFrame mit
    GHI_Wm2/DNI_Wm2/DHI_Wm2 wie von load_era5_weather geliefert) via pvlib:
    Sonnenstand -> Einstrahlung auf geneigte Modulfläche (POA) -> PV-Leistung.
    performance_ratio wird, falls nicht explizit übergeben, aus parameters.yaml
    (system_parameters.PV.performance_ratio) gelesen -- fällt auf 0.85 zurück.
    """
    if performance_ratio is None:
        performance_ratio = (
            _project_parameters().get("system_parameters", {}).get("PV", {}).get("performance_ratio", 0.85)
        )

    # Sonnenstandsberechnung braucht einen eindeutigen (tz-aware) Kalender;
    # UTC ist hierfür ausreichend genau und eindeutig (kein DST-Mehrdeutigkeitsproblem).
    index_utc = pd.DatetimeIndex(weather.index, tz="UTC")

    location = pvlib.location.Location(lat, lon, tz="UTC")
    solpos = location.get_solarposition(index_utc)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos["apparent_zenith"].to_numpy(),
        solar_azimuth=solpos["azimuth"].to_numpy(),
        dni=weather["DNI_Wm2"].to_numpy(),
        ghi=weather["GHI_Wm2"].to_numpy(),
        dhi=weather["DHI_Wm2"].to_numpy(),
    )

    p_pv_mw = (np.asarray(poa["poa_global"]) / 1000.0) * pv_capacity_MW * performance_ratio
    return pd.Series(np.clip(p_pv_mw, 0, None), index=weather.index, name="P_PV_MW")


#-------------------------------------------------------------------------
# Alles in einem: Modell-Inputs für ein Jahr
#-------------------------------------------------------------------------

def load_era5_model_inputs(year=2019, lat=LAT, lon=LON, cache_dir=CACHE_DIR,
                            surface_tilt=35, surface_azimuth=180,
                            pv_capacity_MW=5.0, performance_ratio=None,
                            T_supply_C=None, eta_carnot=None, max_parallel=6):
    """
    Ein Aufruf -> alle ERA5-basierten Modell-Inputs: T_amb_C, wind_speed_ms,
    GHI/DHI/DNI_Wm2, COP_t (temperaturabhängig) und P_PV_MW (eigenes PV-Modell).
    Ersetzt load_temperature_data() + read_pv_data() für das Optimierungsmodell.
    performance_ratio/T_supply_C/eta_carnot: siehe compute_pv_generation()/compute_cop()
    -- per Default aus parameters.yaml gelesen, nicht mehr fest im Code hinterlegt.
    """
    weather = load_era5_weather(year, lat, lon, cache_dir, max_parallel=max_parallel)
    weather["COP_t"]   = compute_cop(weather["T_amb_C"], T_supply_C, eta_carnot)
    weather["P_PV_MW"] = compute_pv_generation(
        weather, lat, lon,
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        pv_capacity_MW=pv_capacity_MW, performance_ratio=performance_ratio,
    )
    return weather


if __name__ == "__main__":
    import argparse
    import sys

    # Robust gegen die Startart (Spyder-"Run file", `python era5_weather.py`,
    # `python -m funcs.era5_weather`): funcs' Elternordner (src/ACES-2026) explizit
    # auf den Importpfad legen, statt uns auf sys.path[0] zu verlassen.
    if str(REPO_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_SRC_DIR))
    from funcs.read_data import read_parameters

    # Anlagenparameter standardmäßig aus parameters.yaml (system_parameters.PV),
    # per CLI überschreibbar. Fällt auf feste Defaults zurück, falls parameters.yaml
    # (noch) keine tilt/azimuth/performance_ratio-Einträge enthält.
    try:
        pv_defaults = read_parameters(str(REPO_SRC_DIR / "parameters.yaml"))["system_parameters"]["PV"]
    except (FileNotFoundError, KeyError):
        pv_defaults = {}

    parser = argparse.ArgumentParser(
        description="ERA5-Wetterdaten (Temperatur, Strahlung, Wind) abrufen, aufbereiten und "
                     "als CSV speichern (inkl. eigenem PV-Ertragsmodell via pvlib). "
                     "Standalone-Skript: python -m funcs.era5_weather --year 2019"
    )
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--lat", type=float, default=LAT)
    parser.add_argument("--lon", type=float, default=LON)
    parser.add_argument("--pv-capacity-mw", type=float,
                         default=pv_defaults.get("initial_pv_capacity", 5.0))
    parser.add_argument("--tilt", type=float,
                         default=pv_defaults.get("surface_tilt", 35))
    parser.add_argument("--azimuth", type=float,
                         default=pv_defaults.get("surface_azimuth", 180))
    parser.add_argument("--performance-ratio", type=float,
                         default=pv_defaults.get("performance_ratio", 0.85))
    parser.add_argument("--supply-temp", type=float, default=None,
                         help="Vorlauftemperatur [°C] für COP_t (default: net_parameters.supply_temperature "
                              "aus parameters.yaml)")
    parser.add_argument("--eta-carnot", type=float, default=None,
                         help="Gütegrad rel. Carnot-Wirkungsgrad für COP_t (default: "
                              "system_parameters.HP.eta_carnot aus parameters.yaml)")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR,
                         help="Verzeichnis für den ERA5-NetCDF-Cache (default: %(default)s)")
    parser.add_argument("--parallel", type=int, default=6,
                         help="Max. gleichzeitige CDS-Anfragen für fehlende Monate "
                              "(default: %(default)s; 1 = sequenziell)")
    parser.add_argument("--out", type=str, default=None,
                         help="Ziel-CSV (default: <cache-dir>/era5_weather_pv_<lat>_<lon>_<year>.csv)")
    args = parser.parse_args()

    df = load_era5_model_inputs(
        year=args.year, lat=args.lat, lon=args.lon, cache_dir=args.cache_dir,
        surface_tilt=args.tilt, surface_azimuth=args.azimuth,
        pv_capacity_MW=args.pv_capacity_mw, performance_ratio=args.performance_ratio,
        T_supply_C=args.supply_temp, eta_carnot=args.eta_carnot,
        max_parallel=args.parallel,
    )

    out_path = args.out or f"{args.cache_dir}/era5_weather_pv_{args.lat:.3f}_{args.lon:.3f}_{args.year}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)

    print(f"\nGespeichert: {out_path}  ({len(df)} Stunden)")
    print(df[["T_amb_C", "COP_t", "GHI_Wm2", "DNI_Wm2", "wind_speed_ms", "P_PV_MW"]].describe())
    print(f"\nPV-Jahresertrag: {df['P_PV_MW'].sum():.0f} MWh  "
          f"| Volllaststunden: {df['P_PV_MW'].sum() / args.pv_capacity_mw:.0f} h")
