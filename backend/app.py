from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests, math

app = FastAPI(title="Railway Network Schematic Backend", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Station list (input)
# -------------------------
STATIONS_INPUT = [
    {"id": "SC", "name": "Secunderabad Jn", "lat": 17.4365, "lon": 78.4983, "platforms": 6},
    {"id": "MJF", "name": "Malkajgiri",      "lat": 17.4521, "lon": 78.5230, "platforms": 2},
    {"id": "BMO", "name": "Bolarum",         "lat": 17.5247, "lon": 78.5238, "platforms": 2},
    {"id": "MDF", "name": "Medchal",         "lat": 17.6013, "lon": 78.6185, "platforms": 3},
]

# -------------------------
# Helper to fetch OSM (can extend later)
# -------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def fetch_osm_network(stations):
    # for now we just return synthetic — you can extend to OSM parsing later
    return {
        "stations": stations,
        "tracks": [], "signals": [], "level_crossings": [], "junctions": []
    }

# -------------------------
# Build schematic layout
# -------------------------
@app.get("/schematic")
def get_schematic(osm: bool = False):
    network = fetch_osm_network(STATIONS_INPUT) if osm else {"stations": STATIONS_INPUT}

    spacing_x = 200   # distance between stations
    offset_x = 100
    offset_y = 200

    schematic_stations = []
    schematic_blocks = []
    schematic_signals = []
    schematic_crossings = []
    schematic_junctions = []

    # Place stations linearly
    for i, st in enumerate(network["stations"]):
        x = offset_x + i * spacing_x
        y = offset_y
        schematic_stations.append({
            "id": st["id"], "name": st["name"], "platforms": st.get("platforms", 1),
            "x": x, "y": y
        })

        # create block between previous and current station
        if i > 0:
            prev = schematic_stations[i-1]
            block_id = f"BLOCK_{prev['id']}_{st['id']}"
            block_x = prev["x"] + 40
            block_y = y - 10
            block_w = spacing_x - 80
            block_h = 20

            schematic_blocks.append({
                "id": block_id, "from": prev["id"], "to": st["id"],
                "status": "free", "x": block_x, "y": block_y,
                "width": block_w, "height": block_h
            })

            # add one signal per block
            schematic_signals.append({
                "id": f"SIGNAL_{block_id}",
                "x": block_x + block_w // 2,
                "y": block_y - 30,
                "aspect": "RED"
            })

            # demo crossing under the middle block
            if i == 2:  # just an example
                schematic_crossings.append({
                    "id": f"CROSS_{block_id}",
                    "x": block_x + block_w // 2,
                    "y": block_y + 60,
                    "status": "open"
                })

    return {
        "stations": schematic_stations,
        "blocks": schematic_blocks,
        "signals": schematic_signals,
        "crossings": schematic_crossings,
        "junctions": schematic_junctions,
        "trains": [],
        "canvas": {
            "width": offset_x + len(STATIONS_INPUT) * spacing_x,
            "height": offset_y + 200
        }
    }

@app.get("/")
def root():
    return {"message": "Railway schematic backend running"}


@app.get("/stations")
def get_stations():
    return {"stations": STATIONS_INPUT}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
