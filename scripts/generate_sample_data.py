import json
import random
import os

def generate_training_data(num_samples=1000):
    data = []
    road_types = ["highway", "arterial", "collector", "residential", "service"]
    road_surfaces = ["asphalt", "concrete", "gravel", "unpaved"]
    
    for i in range(num_samples):
        segment_id = f"SEG_{i:04d}"
        road_type = random.choice(road_types)
        road_width = random.choice([3.5, 5.0, 7.5, 10.0])
        road_surface = random.choice(road_surfaces)
        
        # Continuous values 0-1
        road_condition = random.uniform(0, 1)
        intersection_density = random.uniform(0, 1)
        traffic_level = random.uniform(0, 1)
        congestion = random.uniform(0, 1)
        vehicle_density = random.uniform(0, 1)
        lighting = random.uniform(0, 1)
        pedestrian_density = random.uniform(0, 1)
        weather_risk = random.uniform(0, 1)
        
        # Introduce some missing values or invalid for testing preprocessing
        if random.random() < 0.05:
            road_width = -1.0 # Invalid
        if random.random() < 0.05:
            road_condition = None # Missing
            
        # Target logic (simulated rule for risk score 0-100)
        # High traffic, bad condition, bad weather, bad lighting increase risk
        base_risk = 20
        risk = base_risk
        
        if road_condition is not None:
            risk += (1 - road_condition) * 20
        risk += traffic_level * 15
        risk += congestion * 15
        risk += (1 - lighting) * 10
        risk += weather_risk * 10
        risk += intersection_density * 5
        risk += pedestrian_density * 5
        
        if road_type == "residential":
            risk -= 5
            
        risk = max(0, min(100, int(risk)))
        
        data.append({
            "segment_id": segment_id,
            "road_type": road_type,
            "road_width": road_width,
            "road_surface": road_surface,
            "road_condition": road_condition,
            "intersection_density": intersection_density,
            "traffic_level": traffic_level,
            "congestion": congestion,
            "vehicle_density": vehicle_density,
            "lighting": lighting,
            "pedestrian_density": pedestrian_density,
            "weather_risk": weather_risk,
            "risk_score": risk
        })
        
    return data

def generate_sample_route():
    # As requested in the prompt
    return {
        "route_id": "R001",
        "segments": [
            {
                "segment_id": "SEG_001",
                "road_type": "residential",
                "road_width": 5.5,
                "road_surface": "asphalt",
                "road_condition": 0.8,
                "intersection_density": 0.2,
                "traffic_level": 0.2,
                "congestion": 0.1,
                "vehicle_density": 0.2,
                "lighting": 0.9,
                "pedestrian_density": 0.3,
                "weather_risk": 0.1
            },
            {
                "segment_id": "SEG_002",
                "road_type": "arterial",
                "road_width": 7.5,
                "road_surface": "asphalt",
                "road_condition": 0.6,
                "intersection_density": 0.5,
                "traffic_level": 0.8,
                "congestion": 0.7,
                "vehicle_density": 0.8,
                "lighting": 0.7,
                "pedestrian_density": 0.5,
                "weather_risk": 0.2
            },
            {
                "segment_id": "SEG_003",
                "road_type": "collector",
                "road_width": 5.0,
                "road_surface": "asphalt",
                "road_condition": 0.3,
                "intersection_density": 0.9,
                "traffic_level": 0.6,
                "congestion": 0.5,
                "vehicle_density": 0.6,
                "lighting": 0.5,
                "pedestrian_density": 0.6,
                "weather_risk": 0.3
            },
            {
                "segment_id": "SEG_004",
                "road_type": "arterial",
                "road_width": 7.5,
                "road_surface": "concrete",
                "road_condition": 0.7,
                "intersection_density": 0.6,
                "traffic_level": 0.9,
                "congestion": 0.9,
                "vehicle_density": 0.9,
                "lighting": 0.2,
                "pedestrian_density": 0.8,
                "weather_risk": 0.4
            },
            {
                "segment_id": "SEG_005",
                "road_type": "residential",
                "road_width": 5.5,
                "road_surface": "asphalt",
                "road_condition": 0.9,
                "intersection_density": 0.3,
                "traffic_level": 0.3,
                "congestion": 0.2,
                "vehicle_density": 0.3,
                "lighting": 0.8,
                "pedestrian_density": 0.4,
                "weather_risk": 0.1
            }
        ]
    }

if __name__ == "__main__":
    import pandas as pd
    
    # Generate training data
    train_data = generate_training_data(2000)
    df = pd.DataFrame(train_data)
    
    os.makedirs("data/sample", exist_ok=True)
    df.to_csv("data/sample/training_data.csv", index=False)
    print(f"Generated {len(df)} training samples at data/sample/training_data.csv")
    
    # Generate route data
    route_data = generate_sample_route()
    with open("data/sample/route_data.json", "w") as f:
        json.dump(route_data, f, indent=2)
    print("Generated sample route at data/sample/route_data.json")
