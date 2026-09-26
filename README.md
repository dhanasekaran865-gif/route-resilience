# Route Resilience Prototype

## Project Overview

Route Resilience is an intelligent route-analysis system that goes beyond conventional navigation. It assesses the resilience and risks along a route based on road, traffic, infrastructure, and environmental features. This prototype is the **Data + Perception / Prediction** module, designed to ingest route data, preprocess features, run predictions using a machine learning model, and output an explainable JSON resilience assessment.

## Architecture

Raw Data → Collection → Preprocessing → Feature Extraction → ML → Prediction → Route Risk

## Features

The model analyzes 10-12 key features for each segment:
- **Road:** `road_type`, `road_width`, `road_surface`, `road_condition`, `intersection_density`
- **Traffic:** `traffic_level`, `congestion`, `vehicle_density`
- **Environmental:** `lighting`, `pedestrian_density`, `weather_risk`

## ML Model

- **Algorithm:** Random Forest Regressor
- **Why:** Selected for its interpretability, robustness to outliers, and strong performance without requiring exhaustive tuning, making it ideal for a reliable prototype.
- **Training:** The model is trained to predict a risk score from 0-100.
- **Evaluation:** MAE, RMSE, and R2 scores are logged during training.
- **Limitations:** Currently trained on simulated data. Does not reflect real-world safety risks until retrained on ground-truth data.

## Important Disclaimer: Simulated Data

> **WARNING:** The data and risk labels provided in this repository are **simulated**. This project demonstrates the *pipeline architecture* and the system's ability to process features and expose predictions via API. It does **not** generate accurate real-world road risk predictions until connected to a real data source and trained on valid labels.

## Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

*(Core dependencies: pandas, scikit-learn, fastapi, uvicorn, pydantic)*

### 2. Generate Data & Train Model

First, generate the simulated datasets and train the Random Forest model:

```bash
python scripts/generate_sample_data.py
python ml/training/train.py
```

### 3. Run Pipeline Demo

Run the end-to-end pipeline to see the risk analysis output:

```bash
python scripts/run_pipeline.py
```

### 4. Start the API

Start the FastAPI application:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- **Health Check:** `GET /health`
- **Predict Route:** `POST /predict/route` (Provide a JSON with a route ID and list of segments)

## Future Improvements

The architecture exposes clean interfaces (`DataProvider`, `FeaturePreprocessor`) designed to be extended:
- **Mapping:** Integration with OpenStreetMap or routing APIs.
- **Traffic:** Live traffic sensors or APIs.
- **Environmental:** Weather and flood datasets.
- **Computer Vision:** Street-level imagery processing to automate `road_condition` and `pedestrian_density` extraction.

## Weather API Integration (Phase 3A)

Route Resilience now integrates **Open-Meteo**, a robust, keyless weather API, to fetch live weather metrics based on the route's geographic midpoint. 

### Why Open-Meteo?
It was chosen because it requires no API key, making the prototype instantly usable for development, while offering detailed `temperature`, `precipitation`, `wind_speed`, and `visibility` metrics globally.

### Configuration
Update the environment variables in `.env.example` (or your local `.env`) to switch providers:
```
WEATHER_PROVIDER=open-meteo # or 'mock'
WEATHER_API_KEY= # Not required for open-meteo
```

### Feature Engineering & Caveats
The `weather_risk` feature dynamically scales `[0, 1]` based on severity (e.g., heavy rain > 5mm, wind > 30km/h, poor visibility).
> **IMPORTANT:** Weather risk is a **prototype feature-engineering calculation** and is not a scientifically validated accident-risk probability. Furthermore, the ML predictor is trained on synthetic data, meaning real safety outcomes are not perfectly mapped yet.

### Provenance and Caching
All weather requests use lightweight `(lat, lon, hour)` caching to avoid redundant lookups. Failures degrade gracefully into `"source": "missing"` metadata tags rather than crashing the pipeline, while successful requests propagate `"source": "weather_api"` all the way to the frontend!

## Traffic API Integration (Phase 3B)

Route Resilience now integrates **TomTom Traffic Flow** to compute live road congestion.

### Why TomTom?
TomTom's flow segments distinctly provide `currentSpeed` and `freeFlowSpeed`, giving us the exact variables necessary to calculate a defensible mathematical `congestion` ratio `[0, 1]` rather than fabricating or arbitrarily classifying traffic levels. 

### Configuration
```
TRAFFIC_PROVIDER=tomtom # or 'mock'
TOMTOM_API_KEY=your_key
```

### Feature Engineering & Caveats
- `congestion` is computed as `1.0 - (currentSpeed / freeFlowSpeed)`, properly clamped between 0 and 1.
- `traffic_level` uses threshold boundaries based on the calculated congestion.
- **`vehicle_density` is explicitly marked as `missing`**: TomTom does not count cars natively on this endpoint. In adherence to strict data-integrity rules, this feature is flagged safely as `missing` (`0.0` coverage) and is passed through gracefully to the ML pipeline.

### Caching and Data Integrity
- Midpoint traffic queries are cached within 5-minute buckets.
- Provenance metadata distinguishes derived data (`"status": "derived", "derived_from": ["current_speed", "free_flow_speed"]`).

## Phase 4 — Real-World Safety Ground Truth

Route Resilience introduces an advanced ground-truth safety pipeline designed to transition the prototype away from purely synthetic heuristics toward a scientifically defensible real-data model.

### Why historical safety data?
A model predicting risk based on predefined rules (Phase 1-3) is fundamentally a rules engine. To deploy a true predictive algorithm, the model must be trained on actual documented historical collision data. We ingest official **UK STATS19 collision records** for this purpose.

### Architecture
```text
STATS19 (CSV)
   ↓
Data Cleaning & Geospatial Reprojection
   ↓
OSM Road Matching (KDTree / Spatial Index)
   ↓
Historical Segment Dataset (Positive Matches)
   ↓
Negative Edge Sampling
   ↓
Feature Construction (Static OSM Data ONLY)
   ↓
Train / Validation / Test
   ↓
Random Forest Classifier (`real_safety_model.joblib`)
   ↓
Evaluation Metrics
```

### Data Leakage Prevention
To prevent temporal data leakage, this model explicitly avoids querying real-time traffic and weather APIs for historical crashes. Because applying today's weather/traffic to a 3-year-old crash constitutes severe leakage, these features are forcefully set to `missing` during historical training. Only static structural OSM geometry (`road_type`, `surface`, `width`, `intersection_density`) is used to predict the baseline segment risk. 

### Positive and Negative Sampling
Because STATS19 naturally contains only crash events (Positive samples), the pipeline procedurally extracts equivalent crash-free edges from the OSM graph bounding box (Negative samples). This forces the model to learn structural differentiators rather than just outputting `1` constantly.

### Limitations and Safety Disclaimer
> **IMPORTANT:** The current system is a research prototype. Historical collision data is used as ground truth for model development, but model outputs should not be interpreted as guaranteed accident probabilities or as a substitute for professional road-safety analysis.

- Historical crash data is subject to reporting limitations and underreporting.
- "No-crash" sampled segments do not necessarily indicate zero risk, merely zero reported collisions during the window.
- Traffic and weather alignment are unavailable without historical archives.
- The `real_safety_model.joblib` baseline evaluates only spatial network properties at this stage.

*Note: Phase 1-3's synthetic model remains active and decoupled from Phase 4.*

## Phase 5 — Multi-Factor Route Resilience Scoring

Route Resilience Phase 5 adds a dynamic Resilience Engine capable of combining multiple hazard layers (structural geometry, traffic, weather, baseline historical safety, and live network disruptions) into a multi-factor resilience score.

### Architecture

1. **Common Segment Schema**: Segments seamlessly unify physical features (`road_type`, `road_width`), probabilistic features (`safety_risk`), current flow state (`current_speed`, `free_flow_speed`), and environmental context (`weather_features`).
2. **Safety Predictor Wrapper**: Safely evaluates the historical Phase 4 model on new edges, stripping real-time data automatically to prevent temporal leakage before inference.
3. **Graph Healing Interface**: A strict gatekeeping protocol in `graph_healing.py` requiring explicit topology validation before edges are appended to the routing graph.
4. **Disruption Engine**: Simulates hazard constraints (e.g. `FLOODED`) using extreme edge penalties instead of graph fracturing, maintaining full visibility of detours and remaining alternative pathways.

### Route Resilience Metrics

The Resilience Engine calculates:
* Total travel distance and time under varied scenarios
* Average and Maximum flow congestion
* Safety-risk exposure (from Phase 4 Random Forest output)
* Number of disrupted segments encountered
* Available Edge-Disjoint alternative routes
* Reachability and exact structural Detour Distance in disasters

### Safety Limitations

**This prototype does not guarantee road safety, predict crashes with certainty, or guarantee route availability during disasters.**

The `resilience_score` is a heuristically weighted prototype metric. Missing values intentionally degrade to NaN and rely safely on SimpleImputation, maintaining strict adherence to offline processing without artificially fabricating historical values.

### Running Phase 5

To evaluate the resilience demo:
```bash
python scripts/run_resilience_demo.py
```
To run the full test suite (including Phase 1-5):
```bash
python -m unittest discover tests
```
## Phase 7 — Real Geographic Route Visualization

Phase 7 replaces conceptual frontend vectors with exact GeoJSON `LineString` paths sourced natively from the underlying OpenStreetMap dataset. 

### Implementation Details
* **Geometry Source:** Geometry is natively extracted from the `osmnx` NetworkX graph edge attributes.
* **Coordinate System:** WGS84 / EPSG:4326. Coordinates are strictly encoded as `[longitude, latitude]` for GeoJSON compliance.
* **MultiDiGraph Edge Selection:** The API resolves parallel edges identical to the routing engine (minimizing weight/length) ensuring visual correctness matches routing topology.
* **Frontend Visualization:** The Leaflet map now dynamically ingests `geometry` objects. It draws the primary route, alternative route, and specifically isolates simulated disruptions using distinct markers, avoiding reliance purely on color (using thickness, dashed patterns, and an explicit legend).
* **Missing Data Fallback:** If OSM doesn't specify a curvilinear `geometry` on an edge, the system safely falls back to a straight-line vector between the topological endpoints. The system *does not fabricate geographic coordinates*.

# Route Resilience Project

Smart routing that evaluates how routes perform when traffic and road conditions change.

## Architecture Flow

The system supports taking a real OpenStreetMap bounding box, OR taking a user-uploaded road graph as the topology input.

**With Graph Healing (Phase 8):**
Graph healing predicts plausible missing connections from graph structure; it does not directly reconstruct roads from satellite imagery.

```
London GraphML
→ Real OSM road graph
→ Controlled topology damage
→ Candidate generation
→ ML prediction
→ Validation gate
→ Healed graph
→ Route optimization
→ Traffic/Safety/Weather
→ Flood simulation
→ Resilient routing
```

## Running the Application
1. Start Backend: `python -m uvicorn api.main:app --reload`
2. Start Frontend: `cd frontend && npm run dev`
3. Run the demo script: `python scripts/run_uploaded_graph_demo.py`
