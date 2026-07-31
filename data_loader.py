import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Maharashtra 36 districts with actual geographical regions, approximate population densities,
# average slopes, vulnerability indexes (0.0 to 1.0), and coastal flags.
DISTRICTS_METADATA = {
    # Konkan Coast (High rain, coastal, cyclones, flood, landslide risk)
    "Mumbai City": {"region": "Konkan", "pop_density": 19652, "slope": 2, "vulnerability": 0.85, "coastal": 1},
    "Mumbai Suburban": {"region": "Konkan", "pop_density": 20980, "slope": 3, "vulnerability": 0.80, "coastal": 1},
    "Thane": {"region": "Konkan", "pop_density": 1157, "slope": 5, "vulnerability": 0.75, "coastal": 1},
    "Palghar": {"region": "Konkan", "pop_density": 362, "slope": 8, "vulnerability": 0.70, "coastal": 1},
    "Raigad": {"region": "Konkan", "pop_density": 368, "slope": 12, "vulnerability": 0.75, "coastal": 1},
    "Ratnagiri": {"region": "Konkan", "pop_density": 197, "slope": 18, "vulnerability": 0.80, "coastal": 1},
    "Sindhudurg": {"region": "Konkan", "pop_density": 167, "slope": 20, "vulnerability": 0.75, "coastal": 1},

    # Western Ghats / Pune Division (Heavy rain in ghats, landslide risk, river floods)
    "Pune": {"region": "Pune Division", "pop_density": 603, "slope": 10, "vulnerability": 0.75, "coastal": 0},
    "Satara": {"region": "Pune Division", "pop_density": 287, "slope": 15, "vulnerability": 0.70, "coastal": 0},
    "Kolhapur": {"region": "Pune Division", "pop_density": 504, "slope": 12, "vulnerability": 0.75, "coastal": 0},
    "Sangli": {"region": "Pune Division", "pop_density": 329, "slope": 5, "vulnerability": 0.65, "coastal": 0},
    "Solapur": {"region": "Pune Division", "pop_density": 290, "slope": 2, "vulnerability": 0.70, "coastal": 0},

    # Nashik Division (Mixed terrain, flood, heatwave risk)
    "Nashik": {"region": "Nashik Division", "pop_density": 393, "slope": 8, "vulnerability": 0.70, "coastal": 0},
    "Ahmednagar": {"region": "Nashik Division", "pop_density": 266, "slope": 4, "vulnerability": 0.68, "coastal": 0},
    "Jalgaon": {"region": "Nashik Division", "pop_density": 360, "slope": 3, "vulnerability": 0.65, "coastal": 0},
    "Dhule": {"region": "Nashik Division", "pop_density": 285, "slope": 4, "vulnerability": 0.62, "coastal": 0},
    "Nandurbar": {"region": "Nashik Division", "pop_density": 277, "slope": 10, "vulnerability": 0.72, "coastal": 0},

    # Marathwada (Dry, plateau, drought-prone, heatwave-prone)
    "Aurangabad": {"region": "Marathwada", "pop_density": 366, "slope": 3, "vulnerability": 0.72, "coastal": 0},
    "Jalna": {"region": "Marathwada", "pop_density": 254, "slope": 2, "vulnerability": 0.70, "coastal": 0},
    "Parbhani": {"region": "Marathwada", "pop_density": 295, "slope": 2, "vulnerability": 0.70, "coastal": 0},
    "Hingoli": {"region": "Marathwada", "pop_density": 244, "slope": 3, "vulnerability": 0.68, "coastal": 0},
    "Nanded": {"region": "Marathwada", "pop_density": 319, "slope": 3, "vulnerability": 0.72, "coastal": 0},
    "Beed": {"region": "Marathwada", "pop_density": 242, "slope": 2, "vulnerability": 0.80, "coastal": 0},
    "Latur": {"region": "Marathwada", "pop_density": 343, "slope": 2, "vulnerability": 0.75, "coastal": 0},
    "Osmanabad": {"region": "Marathwada", "pop_density": 219, "slope": 3, "vulnerability": 0.75, "coastal": 0},

    # Vidarbha (High heatwaves, forest land, river floods, droughts)
    "Nagpur": {"region": "Vidarbha", "pop_density": 470, "slope": 3, "vulnerability": 0.70, "coastal": 0},
    "Wardha": {"region": "Vidarbha", "pop_density": 206, "slope": 2, "vulnerability": 0.65, "coastal": 0},
    "Bhandara": {"region": "Vidarbha", "pop_density": 294, "slope": 2, "vulnerability": 0.68, "coastal": 0},
    "Gondia": {"region": "Vidarbha", "pop_density": 253, "slope": 5, "vulnerability": 0.70, "coastal": 0},
    "Chandrapur": {"region": "Vidarbha", "pop_density": 193, "slope": 3, "vulnerability": 0.75, "coastal": 0},
    "Gadchiroli": {"region": "Vidarbha", "pop_density": 74, "slope": 8, "vulnerability": 0.85, "coastal": 0},
    "Amravati": {"region": "Vidarbha", "pop_density": 237, "slope": 4, "vulnerability": 0.68, "coastal": 0},
    "Akola": {"region": "Vidarbha", "pop_density": 320, "slope": 2, "vulnerability": 0.65, "coastal": 0},
    "Washim": {"region": "Vidarbha", "pop_density": 244, "slope": 3, "vulnerability": 0.65, "coastal": 0},
    "Buldhana": {"region": "Vidarbha", "pop_density": 268, "slope": 3, "vulnerability": 0.68, "coastal": 0},
    "Yavatmal": {"region": "Vidarbha", "pop_density": 204, "slope": 4, "vulnerability": 0.72, "coastal": 0},
}

DISASTER_TYPES = ["Flood", "Cyclone", "Landslide", "Heatwave", "Drought"]
RISK_LEVELS = ["Low", "Medium", "High", "Severe"]

def generate_maharashtra_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generates a realistic dataset representing Maharashtra districts, environmental
    measurements, and calculated disaster risk levels using physics-inspired hazard formulas.
    """
    records = []
    districts = list(DISTRICTS_METADATA.keys())

    for _ in range(n_samples):
        district_name = np.random.choice(districts)
        meta = DISTRICTS_METADATA[district_name]
        
        # Determine logical disaster type based on region properties
        # e.g., Coastal region has higher cyclone/flood prob, Marathwada has drought/heatwave
        if meta["region"] == "Konkan":
            disaster_prob = [0.45, 0.25, 0.20, 0.05, 0.05]  # Flood, Cyclone, Landslide, Heat, Drought
        elif meta["region"] == "Marathwada":
            disaster_prob = [0.05, 0.00, 0.02, 0.38, 0.55]
        elif meta["region"] == "Vidarbha":
            disaster_prob = [0.15, 0.00, 0.02, 0.53, 0.30]
        elif meta["region"] == "Pune Division" and meta["slope"] >= 12:
            disaster_prob = [0.35, 0.00, 0.45, 0.10, 0.10]
        else:
            disaster_prob = [0.25, 0.00, 0.10, 0.35, 0.30]
            
        disaster_type = np.random.choice(DISASTER_TYPES, p=disaster_prob)
        
        # Simulate environmental variables conditional on disaster type and district
        # 1. Rainfall (mm/24h)
        if disaster_type in ["Flood", "Cyclone"]:
            rainfall = np.random.gamma(shape=5.0, scale=35.0)  # High rainfall (100 - 300+ mm)
        elif disaster_type == "Landslide":
            rainfall = np.random.gamma(shape=4.0, scale=30.0)  # Moderate-high triggering rain
        elif disaster_type in ["Heatwave", "Drought"]:
            rainfall = np.random.exponential(scale=5.0)  # Low or no rain (0 - 20 mm)
        else:
            rainfall = np.random.gamma(shape=2.0, scale=15.0)
            
        # Add regional scaling
        if meta["region"] == "Konkan":
            rainfall *= 1.4  # Wet coast
        elif meta["region"] == "Marathwada":
            rainfall *= 0.6  # Dry rain shadow
            
        # 2. Wind Speed (km/h)
        if disaster_type == "Cyclone":
            wind_speed = np.random.normal(loc=110, scale=25)
        elif disaster_type == "Flood":
            wind_speed = np.random.normal(loc=35, scale=12)
        else:
            wind_speed = np.random.normal(loc=15, scale=5)
        wind_speed = max(2.0, wind_speed)
        
        # 3. Temperature (°C)
        if disaster_type == "Heatwave":
            temperature = np.random.normal(loc=43, scale=2.5)
        elif disaster_type == "Drought":
            temperature = np.random.normal(loc=38, scale=3.0)
        else:
            temperature = np.random.normal(loc=28, scale=4.0)
            
        if meta["region"] == "Vidarbha" and disaster_type == "Heatwave":
            temperature += 2.0  # Vidarbha has record-breaking heat waves
            
        # 4. Humidity (%)
        if disaster_type == "Heatwave":
            humidity = np.random.uniform(10, 35)  # Dry heat
        elif disaster_type in ["Flood", "Cyclone"]:
            humidity = np.random.uniform(80, 100) # Heavy moisture
        else:
            humidity = np.random.uniform(40, 75)
            
        # 5. Soil Moisture Index (SMI) (0.0 to 1.0)
        if disaster_type == "Drought":
            soil_moisture = np.random.beta(a=1.5, b=5.0)  # Positively skewed to 0
        elif disaster_type in ["Flood", "Landslide"]:
            soil_moisture = np.random.beta(a=5.0, b=1.5)  # Negatively skewed to 1 (saturated)
        else:
            soil_moisture = np.random.uniform(0.3, 0.7)
            
        # 6. River Level (meters above danger level, can be negative)
        if disaster_type == "Flood":
            river_level = np.random.normal(loc=2.5, scale=1.5)
        else:
            river_level = np.random.normal(loc=-1.5, scale=0.8)
            
        # Calculate scientific hazard score based on physics-inspired equations
        hazard_score = 0.0
        
        if disaster_type == "Flood":
            # Rain, saturated soil, and rising river level
            hazard_score = (
                0.40 * (rainfall / 250.0) +
                0.35 * (max(0, river_level + 2) / 6.0) +
                0.15 * soil_moisture +
                0.10 * meta["vulnerability"]
            )
        elif disaster_type == "Cyclone":
            # High winds, coastal amplification, heavy rains
            hazard_score = (
                0.50 * (wind_speed / 150.0) +
                0.30 * (rainfall / 200.0) +
                0.10 * meta["coastal"] +
                0.10 * meta["vulnerability"]
            )
        elif disaster_type == "Landslide":
            # Steep slope, high triggering rainfall, saturated soil
            hazard_score = (
                0.45 * (rainfall / 200.0) +
                0.35 * (meta["slope"] / 25.0) +
                0.10 * soil_moisture +
                0.10 * meta["vulnerability"]
            )
        elif disaster_type == "Heatwave":
            # Extremely high temperatures, low humidity
            hazard_score = (
                0.70 * ((temperature - 30) / 18.0) +
                0.20 * (1.0 - (humidity / 100.0)) +
                0.10 * meta["vulnerability"]
            )
        elif disaster_type == "Drought":
            # Severe soil moisture deficit, low rainfall, sustained heat
            hazard_score = (
                0.45 * (1.0 - soil_moisture) +
                0.35 * (max(0, 100 - rainfall) / 100.0) +
                0.10 * ((temperature - 20) / 25.0) +
                0.10 * meta["vulnerability"]
            )
            
        records.append({
            "district": district_name,
            "region": meta["region"],
            "pop_density": meta["pop_density"],
            "slope": meta["slope"],
            "coastal": meta["coastal"],
            "vulnerability_index": meta["vulnerability"],
            "rainfall": round(rainfall, 1),
            "wind_speed": round(wind_speed, 1),
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "soil_moisture": round(soil_moisture, 2),
            "river_level": round(river_level, 2),
            "disaster_type": disaster_type,
            "hazard_score": round(hazard_score, 4),
        })
        
    df = pd.DataFrame(records)
    # Bin hazard_score into risk levels dynamically to keep classes logically balanced
    df["risk_level"] = pd.qcut(df["hazard_score"], q=[0, 0.30, 0.65, 0.88, 1.0], labels=RISK_LEVELS).astype(str)
    return df

if __name__ == "__main__":
    print("Generating Maharashtra disaster risk dataset...")
    df = generate_maharashtra_dataset()
    out_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
    out_path = os.path.join(out_dir, "dataset_maharashtra.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} samples and saved to {out_path}")
    print("\nDataset Class Distribution:")
    print(df["risk_level"].value_counts())
    print("\nDisaster Type Distribution:")
    print(df["disaster_type"].value_counts())
