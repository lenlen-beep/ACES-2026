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

# TODO: variable soil temperature?

TARGET_CRS = "EPSG:25832"  # UTM Zone 32N
COORD_PRECISION = 0.01   # [m] — round to cm so adjacent line endpoints snap together


def load_network_gpkg(path, layer, target_crs=TARGET_CRS, length_col="Length"):
    """
    Reads a line layer from a GPKG file, splits MultiLineStrings into simple
    LineStrings, recalculates lengths, and returns a GeoDataFrame and a
    NetworkX graph.

    Parameters
    ----------
    path        : path to the .gpkg file
    layer       : layer name
    target_crs  : target CRS for metric length calculation (default: EPSG:25832)
    length_col  : column name for the computed length in metres

    Returns
    -------
    gdf : GeoDataFrame (LineStrings, in target CRS, with recomputed length)
    G   : nx.Graph (edges = pipes, nodes = line endpoints)
    """

    gdf = gpd.read_file(path, layer=layer)
    # Normalize QGIS column names to the names expected in code
    gdf = gdf.rename(columns={
        "Heating_unit":             "Heating Unit",
        "Connection_Load (dummy)":  "Connection Load",
    })
    if "Diameter_mm" in gdf.columns:
        gdf["Diameter"] = gdf["Diameter_mm"] / 1000.0      # mm -> m
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf.to_crs(target_crs)

    # Round coordinates to cm precision → adjacent line endpoints snap together
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: set_precision(geom, COORD_PRECISION))

    gdf[length_col] = gdf.geometry.length  # [m]

    return gdf


def build_graph(gdf, length_col="Length"):
    """
    Builds a NetworkX graph from a GeoDataFrame of LineStrings.

    - Nodes : line endpoints as (x, y) coordinate tuples
    - Edges : one edge per LineString; all GDF columns stored as attributes
    - The 'geometry' attribute holds the Shapely geometry of the pipe

    Parameters
    ----------
    gdf        : GeoDataFrame with (simple) LineStrings in a metric CRS
    length_col : column with pipe length (used as edge weight)

    Returns
    -------
    G : nx.Graph
    """
    G = nx.Graph()

    # 1. Build graph
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        u = geom.coords[0]   # start node (x, y)
        v = geom.coords[-1]  # end node   (x, y)

        attrs = row.drop("geometry").to_dict()
        attrs["geometry"] = geom

        G.add_node(u, x=u[0], y=u[1])
        G.add_node(v, x=v[0], y=v[1])
        G.add_edge(u, v, **attrs)

    # 2. Transfer edge attributes to end nodes (only now, since all degrees are known)
    for u, v, data in G.edges(data=True):
        # Identify leaf nodes (degree 1) — these are the terminal nodes of end edges
        if G.degree[u] == 1:
            end_node = u
        elif G.degree[v] == 1:
            end_node = v
        else:
            continue

        load = data.get("Connection Load")
        if load is not None and not np.isnan(float(load)) and load > 0:
            G.nodes[end_node]["Connection Load"] = load

        # Transfer Heating Unit flag to end node
        if data.get("Heating Unit"):
            G.nodes[end_node]["Heating Unit"] = True

    return G


def test_connectivity(G, snap_tolerance=0.01, export_path=None):
    """
    Checks whether the graph is topologically connected and whether there are
    geometric gaps between nodes that are close but unconnected.

    Parameters
    ----------
    G              : nx.Graph
    snap_tolerance : maximum distance [m] below which two unconnected nodes
                     are reported as a gap (default: 0.5 m)
    export_path    : path to GeoPackage output file — only created when the
                     graph is not connected (e.g. "Data/components.gpkg")
    """
    print("=== Connectivity Test ===")

    # 1. Topological connectivity
    if nx.is_connected(G):
        print(f"  ✓ Graph is connected  "
              f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    else:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        print(f"  ✗ Graph is NOT connected — {len(components)} components:")
        for i, comp in enumerate(components):
            print(f"      [{i+1}] {len(comp)} nodes")

        if export_path is not None:
            rows = []
            for comp_idx, comp_nodes in enumerate(components, start=1):
                subG = G.subgraph(comp_nodes)
                for u, v, data in subG.edges(data=True):
                    geom = data.get("geometry")
                    if geom is not None:
                        rows.append({"Component": comp_idx,
                                     "Nodes":     len(comp_nodes),
                                     "geometry":   geom})
            if rows:
                gdf_comp = gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)
                gdf_comp.to_file(export_path, driver="GPKG", layer="Components")
                print(f"  → GeoPackage saved: {export_path}")
            else:
                print("  → No geometries available, GeoPackage not created.")

    # 2. Geometric gaps: nodes that are close but not connected
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
        print(f"\n  ✗ {len(gaps)} geometric gap(s) < {snap_tolerance} m found:")
        for u, v, d in sorted(gaps, key=lambda x: x[2]):
            print(f"      {u}  ↔  {v}   Δ = {d:.4f} m")
    else:
        print(f"  ✓ No gaps < {snap_tolerance} m between unconnected nodes")

    print("========================")
    return nx.is_connected(G) and len(gaps) == 0


def create_pandapipes_network(G, pn_bar=6.0):
    """
    Creates a pandapipes two-pipe network (supply + return) from the
    NetworkX graph G.

    Parameters
    ----------
    G              : nx.Graph (from build_graph)
    pn_bar         : nominal pressure [bar] for all junctions
    t_supply_k     : supply temperature [K] (default 353.15 K = 80 °C)
    t_return_k     : return temperature [K] (default 328.15 K = 55 °C)
    cp_j_per_kgk   : specific heat capacity of water [J/(kg·K)]

    Returns
    -------
    net : pandapipes.pandapipesNet
    """
    net = pandapipes.create_empty_network(fluid="water")

    t_supply_k = parameters['net_parameters']['supply_temperature']+273.15
    dt_k = parameters['net_parameters']['delta_T']
    t_return_k = parameters['net_parameters']['supply_temperature']+273.15 - dt_k
    cp_j_per_kgk=parameters['net_parameters']['cp']

    # 1. Create junctions and transfer elements per node
    node_to_jidx_VL = {}
    node_to_jidx_RL = {}

    for coord, data in G.nodes(data=True):
        jidx_VL = pandapipes.create_junction(net, pn_bar=pn_bar, tfluid_k=t_supply_k)
        jidx_RL = pandapipes.create_junction(net, pn_bar=pn_bar, tfluid_k=t_return_k)

        node_to_jidx_VL[coord] = jidx_VL
        node_to_jidx_RL[coord] = jidx_RL

        # Building transfer station: Heat Exchanger + Flow Control
        connection_load = data.get("Connection Load")
        if connection_load is not None:
            qext_w = float(connection_load) * 1000.0  # kW → W
            mdot_kg_per_s = qext_w / (cp_j_per_kgk * dt_k)

            # Intermediate junction between HX outlet and FC inlet
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

        # Heat source/transfer station: circulation pump from return to supply
        if data.get("Heating Unit") and not data.get('Connection Load'):
            pandapipes.create_circ_pump_const_pressure(
                net,
                return_junction=jidx_RL,
                flow_junction=jidx_VL,
                p_flow_bar=pn_bar,
                plift_bar=3,
                t_flow_k=t_supply_k,
            )

    # 2. Create pipes: one supply and one return pipe per edge (return direction reversed)
    pipe_geoms = {}   # pipe_index → Shapely geometry
    pipe_pairs = []   # [(supply_idx, return_idx), ...] in edge order

    for u, v, data in G.edges(data=True):
        length_km = data.get("Length", 0.0) / 1000.0
        diameter_m = data.get('Diameter', 0.0223)  # already in m
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
    Time-series simulation: for each timestamp the building loads are written
    to the Heat Exchangers and Flow Controls, then pipeflow is executed.

    Parameters
    ----------
    net           : pandapipes network (from create_pandapipes_network)
    buildings_df  : DataFrame with a 'Datum' column and one load column per
                    building [kW]
    building_cols : list of load columns (default: all except 'Datum')

    Returns
    -------
    DataFrame with columns 'Datum' and 'mdot_kg_per_s' (pump mass flow)
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
            f"{len(building_cols)} building columns, but {len(hx_idx)} heat exchangers in network."
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
            # t_from_k = return temperature (suction side), t_to_k = supply temperature (pressure side)
            t_return_k = float(pump_row['t_from_k'])
            t_supply_k = float(pump_row['t_to_k'])
        except Exception as e:
            if i == 0:
                print(f"  pipeflow error (timestep 0): {e}")
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
            print(f"  {i + 1}/{n} timesteps simulated ...")

    return pd.DataFrame(records)


def export_res_pipe_gpkg(net, pipe_geoms, pipe_pairs, path, crs=TARGET_CRS):
    """
    Exports net.res_pipe as a GeoPackage with supply and return results as
    separate columns. One row = one pipe (edge), columns: VL_*, RL_*.
    """

    res = net.res_pipe.copy()
    res['specific_p_loss_pa_per_m'] = (
        (res['p_from_bar'] - res['p_to_bar']) / (net.pipe['length_km'] * 1000) * 1e5
    )
    # Add DN from net.pipe (populated after dimension_pipes)
    if 'DN' in net.pipe.columns:
        res['DN'] = net.pipe['DN']

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
    print(f"GeoPackage saved: {path}")


# DN catalogue: (label, inner diameter [m])
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
    """Darcy-Weisbach pressure loss [Pa/m] for a given mass flow and inner diameter.

    Friction factor via Swamee-Jain (explicit approximation of Colebrook-White).
    Water properties at ~70 °C: rho=978 kg/m³, mu=4.04e-4 Pa·s.
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


def fix_pipe_orientations(net):
    """Flips pipes with negative mass flow so that from_junction always
    is the inflow node. Must be called after a pipeflow run so that
    net.res_pipe.mdot_from_kg_per_s is populated.

    Returns: number of pipes flipped.
    """
    n_flipped = 0
    for idx in net.pipe.index:
        mdot = float(net.res_pipe.at[idx, 'mdot_from_kg_per_s'])
        if mdot < 0:
            from_j = net.pipe.at[idx, 'from_junction']
            to_j   = net.pipe.at[idx, 'to_junction']
            net.pipe.at[idx, 'from_junction'] = to_j
            net.pipe.at[idx, 'to_junction']   = from_j
            n_flipped += 1
    if n_flipped:
        print(f"fix_pipe_orientations: {n_flipped} pipe(s) flipped.")
    else:
        print("fix_pipe_orientations: All pipes correctly oriented.")
    return n_flipped


def dimension_pipes(net, parameters):
    """Dimensions all pipes in the pandapipes network using the DN catalogue.

    For each pipe, starting from DN 20, the smallest DN is chosen for which
    the specific pressure loss is ≤ parameters['pipe_parameters']['specific_pressure_loss']
    (Pa/m). Mass flow values come from net.res_pipe (must be populated before calling
    this function, e.g. via a peak-load simulation).

    net.pipe['diameter_m'] is updated in-place.

    Returns
    -------
    DataFrame with columns: pipe_idx, mdot_kg_per_s, DN, diameter_m, dp_pa_per_m
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
            # Mass flow exceeds the limit even at DN 250
            chosen_dn, chosen_d = _DN_CATALOG[-1]
            chosen_dp = _specific_pressure_loss(mdot, chosen_d, kr_m)
            print(f"  WARNING pipe {idx}: {chosen_dp:.0f} Pa/m at {chosen_dn} — limit exceeded!")

        # Back-calculate u_value [W/(m·K)] from old alpha and old diameter,
        # then recompute alpha for the new diameter (alpha = u_value / (π·D))
        old_d     = net.pipe.at[idx, 'diameter_m']
        old_alpha = net.pipe.at[idx, 'alpha_w_per_m2k']
        u_value   = old_alpha * np.pi * old_d
        new_alpha = u_value / (np.pi * chosen_d)

        net.pipe.at[idx, 'diameter_m']      = chosen_d
        net.pipe.at[idx, 'alpha_w_per_m2k'] = new_alpha
        net.pipe.at[idx, 'DN']             = chosen_dn

        rows.append({
            'pipe_idx':        idx,
            'mdot_kg_per_s':   mdot,
            'DN':              chosen_dn,
            'diameter_m':      chosen_d,
            'dp_pa_per_m':     chosen_dp,
        })

    df = pd.DataFrame(rows)
    print(f"\nPipe dimensioning complete ({len(df)} pipes):")
    print(df.groupby('DN').size().rename('Count').to_string())
    return df
