import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pandapipes
from shapely import set_precision
from shapely.geometry import LineString

try:
    from read_data import read_parameters
except ImportError:
    from funcs.read_data import read_parameters
parameters = read_parameters("src/ACES-2026/parameters.yaml")

# TODO: Bodentemperatur variabel?

TARGET_CRS = "EPSG:25832"  # UTM Zone 32N
COORD_PRECISION = 0.01   # [m] — auf cm runden, damit Endpunkte topologisch übereinstimmen


def load_network_gpkg(path, layer, target_crs=TARGET_CRS, length_col="Length"):
    """
    Liest einen Linien-Layer aus einer GPKG-Datei, splittet MultiLineStrings
    in einfache LineStrings, berechnet die Länge neu und gibt ein
    GeoDataFrame sowie einen NetworkX-Graphen zurück.

    Parameter
    ----------
    path        : Pfad zur .gpkg-Datei
    layer       : Layer-Name
    target_crs  : Ziel-KRS für metrische Längenberechnung (Default: EPSG:25832)
    length_col  : Spaltenname für die berechnete Länge in Metern

    Rückgabe
    --------
    gdf : GeoDataFrame (LineStrings, im Ziel-KRS, mit neu berechneter Länge)
    G   : nx.Graph (Kanten = Leitungen, Knoten = Endpunkte der Linien)
    """

    gdf = gpd.read_file(path, layer=layer)
    # Spaltenschema der QGIS-Datei auf die im Code erwarteten Namen normalisieren
    gdf = gdf.rename(columns={
        "Heating_unit":             "Heating Unit",
        "Connection_Load (dummy)":  "Connection Load",
    })
    if "Diameter_mm" in gdf.columns:
        gdf["Diameter"] = gdf["Diameter_mm"] / 1000.0      # mm -> m
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf.to_crs(target_crs)

    # Koordinaten auf cm-Genauigkeit runden → Endpunkte benachbarter Linien rasten ein
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: set_precision(geom, COORD_PRECISION))

    gdf[length_col] = gdf.geometry.length  # [m]

    print(gdf)

    return gdf


def build_graph(gdf, length_col="Length"):
    """
    Erstellt einen NetworkX-Graphen aus einem GeoDataFrame mit LineStrings.

    - Knoten  : Endpunkte der Linien als (x, y)-Koordinatentupel
    - Kanten  : Eine Kante pro LineString, alle GDF-Spalten als Attribute
    - Attribut 'geometry' enthält die Shapely-Geometrie der Leitung

    Parameter
    ----------
    gdf        : GeoDataFrame mit (einfachen) LineStrings im metrischen KRS
    length_col : Spalte mit der Leitungslänge (für Kantengewicht)

    Rückgabe
    --------
    G : nx.Graph
    """
    G = nx.Graph()

    # 1. Graph vollständig aufbauen
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        u = geom.coords[0]   # Startknoten (x, y)
        v = geom.coords[-1]  # Endknoten   (x, y)

        attrs = row.drop("geometry").to_dict()
        attrs["geometry"] = geom

        G.add_node(u, x=u[0], y=u[1])
        G.add_node(v, x=v[0], y=v[1])
        G.add_edge(u, v, **attrs)

    # 2. Kantenattribute auf Endknoten übertragen (erst hier, da jetzt alle Grade bekannt sind)
    for u, v, data in G.edges(data=True):
        # Blattknoten (Grad 1) bestimmen — das ist der Endknoten der Endkante
        if G.degree[u] == 1:
            end_node = u
        elif G.degree[v] == 1:
            end_node = v
        else:
            continue 

        load = data.get("Connection Load")
        if load is not None and not np.isnan(float(load)) and load > 0:
            G.nodes[end_node]["Connection Load"] = load

        # Heating Unit (Hz) auf Endknoten übertragen
        if data.get("Heating Unit"):
            G.nodes[end_node]["Heating Unit"] = True

    return G


def test_connectivity(G, snap_tolerance=0.01, export_path=None):
    """
    Prüft, ob der Graph topologisch verbunden ist und ob es geometrische
    Lücken zwischen nahe beieinanderliegenden Knoten gibt.

    Parameter
    ----------
    G              : nx.Graph
    snap_tolerance : maximaler Abstand [m] unter dem zwei unverbundene
                     Knoten als Lücke gemeldet werden (Default: 0.5 m)
    export_path    : Pfad zur GeoPackage-Ausgabedatei — wird nur erzeugt,
                     wenn der Graph nicht verbunden ist (z.B. "Data/komponenten.gpkg")
    """
    print("=== Connectivity-Test ===")

    # 1. Topologische Verbundenheit
    if nx.is_connected(G):
        print(f"  ✓ Graph ist verbunden  "
              f"({G.number_of_nodes()} Knoten, {G.number_of_edges()} Kanten)")
    else:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        print(f"  ✗ Graph ist NICHT verbunden — {len(components)} Komponenten:")
        for i, comp in enumerate(components):
            print(f"      [{i+1}] {len(comp)} Knoten")

        if export_path is not None:
            rows = []
            for comp_idx, comp_nodes in enumerate(components, start=1):
                subG = G.subgraph(comp_nodes)
                for u, v, data in subG.edges(data=True):
                    geom = data.get("geometry")
                    if geom is not None:
                        rows.append({"Komponente": comp_idx,
                                     "Knoten":     len(comp_nodes),
                                     "geometry":   geom})
            if rows:
                gdf_comp = gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)
                gdf_comp.to_file(export_path, driver="GPKG", layer="Komponenten")
                print(f"  → GeoPackage gespeichert: {export_path}")
            else:
                print("  → Keine Geometrien vorhanden, GeoPackage nicht erzeugt.")

    # 2. Geometrische Lücken: Knoten die nah beieinander liegen, aber nicht verbunden sind
    nodes = list(G.nodes)
    gaps = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            if G.has_edge(ni, nj):
                continue
            dist = np.hypot(ni[0] - nj[0], ni[1] - nj[1])
            if dist < snap_tolerance:
                gaps.append((ni, nj, dist))

    if gaps:
        print(f"\n  ✗ {len(gaps)} geometrische Lücke(n) < {snap_tolerance} m gefunden:")
        for u, v, d in sorted(gaps, key=lambda x: x[2]):
            print(f"      {u}  ↔  {v}   Δ = {d:.4f} m")
    else:
        print(f"  ✓ Keine Lücken < {snap_tolerance} m zwischen unverbundenen Knoten")

    print("========================")
    return nx.is_connected(G) and len(gaps) == 0


def create_pandapipes_network(G, pn_bar=6.0):
    """
    Erstellt ein pandapipes-Zweileitungsnetz (Vorlauf + Rücklauf) aus dem
    NetworkX-Graphen G.

    Parameter
    ----------
    G              : nx.Graph (aus build_graph)
    pn_bar         : Nenndruck [bar] für alle Junctions
    t_supply_k     : Vorlauftemperatur [K] (Default 353.15 K = 80 °C)
    t_return_k     : Rücklauftemperatur [K] (Default 328.15 K = 55 °C)
    cp_j_per_kgk   : spez. Wärmekapazität Wasser [J/(kg·K)]

    Rückgabe
    --------
    net : pandapipes.pandapipesNet
    """
    net = pandapipes.create_empty_network(fluid="water")
    
    t_supply_k = parameters['net_parameters']['supply_temperature']+273.15
    dt_k = parameters['net_parameters']['delta_T']
    t_return_k = parameters['net_parameters']['supply_temperature']+273.15 - dt_k
    cp_j_per_kgk=parameters['net_parameters']['cp']

    # 1. Junctions + Übergabeelemente je Knoten
    node_to_jidx_VL = {}
    node_to_jidx_RL = {}

    for coord, data in G.nodes(data=True):
        jidx_VL = pandapipes.create_junction(net, pn_bar=pn_bar, tfluid_k=t_supply_k)
        jidx_RL = pandapipes.create_junction(net, pn_bar=pn_bar, tfluid_k=t_return_k)

        node_to_jidx_VL[coord] = jidx_VL
        node_to_jidx_RL[coord] = jidx_RL

        # Hausübergabestation: Heat Exchanger + Flow Control
        connection_load = data.get("Connection Load")
        if connection_load is not None:
            qext_w = float(connection_load) * 1000.0  # kW → W
            mdot_kg_per_s = qext_w / (cp_j_per_kgk * dt_k)

            # Verbindungs-Junction zwischen HX-Ausgang und FC-Eingang
            jidx_conn = pandapipes.create_junction(
                net, pn_bar=pn_bar, tfluid_k=t_return_k
            )
            pandapipes.create_heat_exchanger(
                net,
                from_junction=jidx_VL,
                to_junction=jidx_conn,
                qext_w=qext_w,
            )
            pandapipes.create_flow_control(
                net,
                from_junction=jidx_conn,
                to_junction=jidx_RL,
                controlled_mdot_kg_per_s=mdot_kg_per_s,
            )

        # Wärmeerzeuger-/Übergabestation: Umlaufpumpe von RL nach VL
        if data.get("Heating Unit") and not data.get('Connection Load'):
            pandapipes.create_circ_pump_const_pressure(
                net,
                return_junction=jidx_RL,
                flow_junction=jidx_VL,
                p_flow_bar=pn_bar,
                plift_bar=3,
                t_flow_k=t_supply_k,
            )

    # 2. Pipes anlegen: je Kante eine VL- und eine RL-Pipe (RL-Richtung umgekehrt)
    pipe_geoms = {}   # pipe_index → Shapely-Geometrie
    pipe_pairs = []   # [(vl_idx, rl_idx), ...] in Kantenreihenfolge

    for u, v, data in G.edges(data=True):
        length_km = data.get("Length", 0.0) / 1000.0
        diameter_m = data.get('Diameter', 0.0223)  # bereits in m
        u_value = data.get('U-value_W/Km', 0.119)
        geom = data.get("geometry")

        alpha = u_value / (np.pi * diameter_m)

        idx_vl = pandapipes.create_pipe_from_parameters(
            net,
            from_junction=node_to_jidx_VL[u],
            to_junction=node_to_jidx_VL[v],
            length_km=length_km,
            alpha_w_per_m2k=alpha,
            text_k=283.15,
            diameter_m=diameter_m,
            k_mm=parameters['pipe_parameters']['kr'],
        )
        idx_rl = pandapipes.create_pipe_from_parameters(
            net,
            from_junction=node_to_jidx_RL[v],
            to_junction=node_to_jidx_RL[u],
            length_km=length_km,
            alpha_w_per_m2k=alpha,
            text_k=283.15,
            diameter_m=diameter_m,
            k_mm=parameters['pipe_parameters']['kr'],
        )

        pipe_pairs.append((idx_vl, idx_rl))
        if geom is not None:
            pipe_geoms[idx_vl] = geom
            pipe_geoms[idx_rl] = LineString(list(reversed(list(geom.coords))))

    return net, pipe_geoms, pipe_pairs


def run_timeseries(net, buildings_df, building_cols=None):
    """
    Zeitreihensimulation: für jeden Zeitstempel werden die Gebäudelasten
    in Heat Exchanger und Flow Control geschrieben und pipeflow ausgeführt.

    Parameter
    ----------
    net           : pandapipes-Netz (aus create_pandapipes_network)
    buildings_df  : DataFrame mit Spalte 'Datum' und je einer Lastspalte
                    pro Gebäude [kW]
    building_cols : Liste der Lastspalten (default: alle außer 'Datum')

    Rückgabe
    --------
    DataFrame mit Spalten 'Datum' und 'mdot_kg_per_s' (Pumpenmassenstrom)
    """
    import pandas as pd

    cp    = parameters['net_parameters']['cp']
    dt_k  = parameters['net_parameters']['delta_T']

    if building_cols is None:
        building_cols = [c for c in buildings_df.columns if c != 'Datum']

    hx_idx = net.heat_exchanger.index.tolist()
    fc_idx = net.flow_control.index.tolist()

    if len(building_cols) != len(hx_idx):
        raise ValueError(
            f"{len(building_cols)} Gebäudespalten, aber {len(hx_idx)} Heat Exchanger im Netz."
        )

    records = []
    n = len(buildings_df)

    for i, (_, row) in enumerate(buildings_df.iterrows()):
        for j, col in enumerate(building_cols):
            qext_w = float(row[col]) * 1000.0          # kW → W
            mdot   = qext_w / (cp * dt_k)
            net.heat_exchanger.at[hx_idx[j], 'qext_w']                          = qext_w
            net.flow_control.at[fc_idx[j],   'controlled_mdot_kg_per_s']        = mdot

        try:
            pandapipes.pipeflow(net, mode="sequential")
            pump_row  = net.res_circ_pump_pressure.iloc[0]
            pump_mdot = abs(float(net.res_circ_pump_pressure['mdot_from_kg_per_s'].sum()))
            # t_from_k = Rücklauftemperatur (Ansaugseite), t_to_k = Vorlauftemperatur (Druckseite)
            t_return_k = float(pump_row['t_from_k'])
            t_supply_k = float(pump_row['t_to_k'])
        except Exception as e:
            if i == 0:
                print(f"  pipeflow Fehler (Zeitschritt 0): {e}")
            pump_mdot  = float('nan')
            t_return_k = float('nan')
            t_supply_k = float('nan')

        records.append({
            'Datum':         row['Datum'],
            'mdot_kg_per_s': pump_mdot,
            't_return_k':    t_return_k,
            't_supply_k':    t_supply_k,
        })

        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n} Zeitschritte simuliert …")

    return pd.DataFrame(records)


def export_res_pipe_gpkg(net, pipe_geoms, pipe_pairs, path, crs=TARGET_CRS):
    """
    Exportiert net.res_pipe als GeoPackage mit VL- und RL-Ergebnissen als getrennte Spalten.
    Eine Zeile = eine Leitung (Kante), Spalten: VL_*, RL_*.
    """

    res = net.res_pipe.copy()
    res['specific_p_loss_pa_per_m'] = (
        (res['p_from_bar'] - res['p_to_bar']) / (net.pipe['length_km'] * 1000) * 1e5
    )

    rows = []
    for vl_idx, rl_idx in pipe_pairs:
        vl = res.loc[vl_idx].add_prefix('VL_')
        rl = res.loc[rl_idx].add_prefix('RL_')
        row = pd.concat([vl, rl])
        row['geometry'] = pipe_geoms.get(vl_idx)
        rows.append(row)

    gdf_out = gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)
    gdf_out.index = range(len(gdf_out))
    gdf_out.to_file(path, driver='GPKG')
    print(f"GeoPackage gespeichert: {path}")


# DN-Katalog: (Bezeichnung, Innendurchmesser [m])
_DN_CATALOG = [
    ("DN 20",  0.0217),
    ("DN 25",  0.0285),
    ("DN 32",  0.0372),
    ("DN 40",  0.0431),
    ("DN 50",  0.0545),
    ("DN 65",  0.0703),
    ("DN 80",  0.0825),
    ("DN 100", 0.1071),
    ("DN 125", 0.1325),
    ("DN 150", 0.1603),
    ("DN 200", 0.2101),
    ("DN 250", 0.2630),
]


def _specific_pressure_loss(mdot_kg_per_s, d_m, kr_m, rho=978.0, mu=4.04e-4):
    """Darcy-Weisbach Druckverlust [Pa/m] für gegebenen Massenstrom und Innendurchmesser.

    Reibungszahl nach Swamee-Jain (explizite Näherung der Colebrook-White-Gleichung).
    Wasserparameter bei ~70 °C: rho=978 kg/m³, mu=4.04e-4 Pa·s.
    """
    if mdot_kg_per_s <= 0:
        return 0.0
    A  = np.pi / 4 * d_m ** 2
    v  = mdot_kg_per_s / (rho * A)
    Re = rho * v * d_m / mu
    if Re < 1:
        return 0.0
    # Swamee-Jain
    lam = 0.25 / (np.log10(kr_m / (3.7 * d_m) + 5.74 / Re ** 0.9)) ** 2
    return lam * rho * v ** 2 / (2 * d_m)


def dimension_pipes(net, parameters):
    """Dimensioniert alle Rohre im pandapipes-Netz stufenweise nach DN-Katalog.

    Für jedes Rohr wird ab DN 20 aufsteigend das kleinste DN gewählt, bei dem
    der spezifische Druckverlust ≤ parameters['pipe_parameters']['specific_pressure_loss']
    (Pa/m) liegt. Die Massenstromwerte kommen aus net.res_pipe (muss vor dem Aufruf
    durch pandapipes.pipeflow o.ä. befüllt sein, z.B. mit Spitzenlastsimulation).

    net.pipe['diameter_m'] wird in-place aktualisiert.

    Rückgabe
    --------
    DataFrame mit Spalten: pipe_idx, mdot_kg_per_s, DN, diameter_m, dp_pa_per_m
    """
    pp = parameters['pipe_parameters']
    kr_m   = pp['kr'] / 1000                       # mm → m
    dp_max = pp['specific_pressure_loss']           # Pa/m

    rows = []
    for idx in net.pipe.index:
        mdot = abs(float(net.res_pipe.at[idx, 'mdot_from_kg_per_s']))

        chosen_dn, chosen_d, chosen_dp = None, None, None
        for dn_name, d in _DN_CATALOG:
            dp = _specific_pressure_loss(mdot, d, kr_m)
            if dp <= dp_max:
                chosen_dn, chosen_d, chosen_dp = dn_name, d, dp
                break

        if chosen_dn is None:
            # Massenstrom übersteigt auch bei DN 250 das Limit
            chosen_dn, chosen_d = _DN_CATALOG[-1]
            chosen_dp = _specific_pressure_loss(mdot, chosen_d, kr_m)
            print(f"  WARNUNG Rohr {idx}: {chosen_dp:.0f} Pa/m bei {chosen_dn} – Limit überschritten!")

        # u_value [W/(m·K)] aus altem alpha und altem Durchmesser zurückrechnen,
        # dann alpha für neuen Durchmesser neu setzen (alpha = u_value / (π·D))
        old_d     = net.pipe.at[idx, 'diameter_m']
        old_alpha = net.pipe.at[idx, 'alpha_w_per_m2k']
        u_value   = old_alpha * np.pi * old_d
        new_alpha = u_value / (np.pi * chosen_d)

        net.pipe.at[idx, 'diameter_m']      = chosen_d
        net.pipe.at[idx, 'alpha_w_per_m2k'] = new_alpha

        rows.append({
            'pipe_idx':        idx,
            'mdot_kg_per_s':   mdot,
            'DN':              chosen_dn,
            'diameter_m':      chosen_d,
            'dp_pa_per_m':     chosen_dp,
        })

    df = pd.DataFrame(rows)
    print(f"\nRohrdimensionierung abgeschlossen ({len(df)} Rohre):")
    print(df.groupby('DN').size().rename('Anzahl').to_string())
    return df
