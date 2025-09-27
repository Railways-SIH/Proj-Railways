import json
import networkx as nx
import math
from collections import defaultdict

# Load network.json
try:
    with open("network.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: network.json not found.")
    exit(1)

nodes = {n["id"]: n for n in data["nodes"]}
edges = data["edges"]

# Build undirected graph
G = nx.Graph()
for nid, n in nodes.items():
    G.add_node(nid, **n)
for e in edges:
    if e["u"] in nodes and e["v"] in nodes:
        u_node, v_node = nodes[e["u"]], nodes[e["v"]]
        if "lat" in u_node and "lon" in u_node and "lat" in v_node and "lon" in v_node:
            lat1, lon1 = math.radians(u_node["lat"]), math.radians(u_node["lon"])
            lat2, lon2 = math.radians(v_node["lat"]), math.radians(v_node["lon"])
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = 6371 * c
            weight = int(distance * 10)
        else:
            weight = 5
        G.add_edge(e["u"], e["v"], weight=weight)

# Identify significant nodes
def is_significant(nid):
    if nid not in nodes:
        return False
    kind = nodes[nid].get("kind")
    return kind in ("station", "halt", "level_crossing", "crossing") or G.degree[nid] >= 3

# Create track sections
stations, junctions, crossings, blocks = [], [], [], []
block_counter = 1
visited_edges = set()

# Stations
for nid, n in nodes.items():
    if n.get("kind") in ("station", "halt"):
        stations.append({
            "id": f"STN_{nid}",
            "orig_id": nid,
            "type": "station",
            "name": n.get("name", f"Station_{nid}"),
            "platforms": int(n.get("platforms", 2))
        })

# Junctions
for nid in G.nodes():
    if G.degree[nid] >= 3:
        junctions.append({"id": f"JUNC_{nid}", "orig_id": nid, "type": "junction"})

# Crossings
for nid, n in nodes.items():
    if n.get("kind") in ("level_crossing", "crossing"):
        crossings.append({"id": f"CR_{nid}", "orig_id": nid, "type": "crossing"})

# Blocks
for u, v in G.edges():
    edge_key = tuple(sorted([u, v]))
    if edge_key in visited_edges:
        continue
    visited_edges.add(edge_key)
    path = [u, v]
    current, prev = v, u
    while not is_significant(current) and G.degree[current] == 2:
        neighbors = [n for n in G.neighbors(current) if n != prev]
        if not neighbors:
            break
        prev, current = current, neighbors[0]
        path.append(current)
    if is_significant(path[0]) and is_significant(path[-1]) and len(path) > 1:
        weight = sum(G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1))
        blocks.append({
            "id": f"BLOCK_{block_counter}",
            "from": path[0],
            "to": path[-1],
            "type": "block",
            "weight": weight
        })
        block_counter += 1

# Layout calculations
num_stations = len(stations)
width = max(1200, num_stations * 140)
height = 800
margin = 80

lons = [n["lon"] for n in nodes.values() if "lon" in n]
lats = [n["lat"] for n in nodes.values() if "lat" in n]
min_lon, max_lon = min(lons or [0]), max(lons or [0])
min_lat, max_lat = min(lats or [0]), max(lats or [0])

def scale_x(lon):
    if max_lon == min_lon:
        return margin + width / 2
    return margin + ((lon - min_lon) / (max_lon - min_lon)) * (width - 2 * margin)

def scale_y(lat):
    if max_lat == min_lat:
        return height / 2
    return height - margin - ((lat - min_lat) / (max_lat - min_lat)) * (height - 2 * margin)

coords = {}
stations_sorted = sorted(stations, key=lambda s: nodes.get(s["orig_id"], {}).get("lon", 0))
for i, s in enumerate(stations_sorted):
    nid = s["orig_id"]
    lon = nodes.get(nid, {}).get("lon", min_lon + i * (max_lon - min_lon) / max(1, len(stations)-1))
    lat = nodes.get(nid, {}).get("lat", 0)
    x = scale_x(lon)
    y = scale_y(lat) if abs(max_lat - min_lat) > 0.01 else 120
    s.update({"x": float(x), "y": float(y), "width": 80, "height": 12})
    coords[nid] = (x, y)

for j in junctions:
    nid = j["orig_id"]
    neigh_lons = [nodes.get(n, {}).get("lon", 0) for n in G.neighbors(nid) if "lon" in nodes.get(n, {})]
    lon = sum(neigh_lons) / len(neigh_lons) if neigh_lons else min_lon
    neigh_lats = [nodes.get(n, {}).get("lat", 0) for n in G.neighbors(nid) if "lat" in nodes.get(n, {})]
    lat = sum(neigh_lats) / len(neigh_lats) if neigh_lats else 0
    x = scale_x(lon)
    y = scale_y(lat) + 60
    j.update({"x": float(x), "y": float(y), "width": 40, "height": 8})
    coords[nid] = (x, y)

for c in crossings:
    nid = c["orig_id"]
    lon = nodes.get(nid, {}).get("lon", min_lon)
    lat = nodes.get(nid, {}).get("lat", 0)
    x = scale_x(lon)
    y = scale_y(lat) + 120
    c.update({"x": float(x), "y": float(y), "width": 30, "height": 6})
    coords[nid] = (x, y)

# Connections and blocks
connections = []
id_map = {s["orig_id"]: s["id"] for s in stations + junctions + crossings}
for b in blocks:
    u, v = b["from"], b["to"]
    ux, uy = coords.get(u, (scale_x(nodes.get(u, {}).get("lon", min_lon)), 120))
    vx, vy = coords.get(v, (scale_x(nodes.get(v, {}).get("lon", min_lon)), 120))
    path = f"M{ux + 40},{uy + 6} L{vx - 40},{vy + 6}"
    bx = (ux + vx) / 2
    by = (uy + vy) / 2
    b.update({"x": float(bx - 40), "y": float(by - 6), "width": 80, "height": 12, "path": path})
    connections.append({
        "from": id_map.get(u, u),
        "to": id_map.get(v, v),
        "type": "main" if any(s["orig_id"] in (u, v) for s in stations) else "branch",
        "path": path
    })

# Deduplicate connections
seen = set()
dedup_connections = [c for c in connections if tuple(sorted([c["from"], c["to"]])) not in seen and not seen.add(tuple(sorted([c["from"], c["to"]])))]
track_sections = stations + blocks + junctions + crossings

# Build GRAPH for backend
GRAPH = defaultdict(dict)
for b in blocks:
    u_id = id_map.get(b["from"], b["from"])
    v_id = id_map.get(b["to"], b["to"])
    w = b.get("weight", 5)
    GRAPH[u_id][v_id] = w
    GRAPH[v_id][u_id] = w
for u, v, edata in G.edges(data=True):
    if is_significant(u) and is_significant(v):
        u_id = id_map.get(u, f"UNK_{u}")
        v_id = id_map.get(v, f"UNK_{v}")
        if v_id not in GRAPH[u_id]:
            GRAPH[u_id][v_id] = edata.get("weight", 5)
            GRAPH[v_id][u_id] = edata.get("weight", 5)

# Save layout
layout = {
    "TRACK_SECTIONS": track_sections,
    "CONNECTIONS": dedup_connections,
    "GRAPH": dict(GRAPH)
}
with open("schematic_layout.json", "w") as f:
    json.dump(layout, f, indent=2)

print("schematic_layout.json generated with GRAPH.")