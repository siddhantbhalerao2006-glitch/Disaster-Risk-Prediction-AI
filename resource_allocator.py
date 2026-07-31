import numpy as np
import math

# Base resource requirements per disaster type for a standard baseline region
BASE_RESOURCES = {
    "Flood": {
        "Rescue Boats": 8,
        "Medical Kits": 150,
        "Food Packets": 1200,
        "Emergency Personnel": 45,
        "Shelter Tents": 100,
        "Water Tankers": 2,
        "Cooling Center Kits": 0
    },
    "Cyclone": {
        "Rescue Boats": 5,
        "Medical Kits": 200,
        "Food Packets": 1500,
        "Emergency Personnel": 60,
        "Shelter Tents": 150,
        "Water Tankers": 3,
        "Cooling Center Kits": 0
    },
    "Landslide": {
        "Rescue Boats": 0,
        "Medical Kits": 100,
        "Food Packets": 500,
        "Emergency Personnel": 35,
        "Shelter Tents": 60,
        "Water Tankers": 1,
        "Cooling Center Kits": 0
    },
    "Heatwave": {
        "Rescue Boats": 0,
        "Medical Kits": 180,
        "Food Packets": 0,
        "Emergency Personnel": 20,
        "Shelter Tents": 0,
        "Water Tankers": 8,
        "Cooling Center Kits": 15
    },
    "Drought": {
        "Rescue Boats": 0,
        "Medical Kits": 80,
        "Food Packets": 1000,
        "Emergency Personnel": 15,
        "Shelter Tents": 0,
        "Water Tankers": 30,
        "Cooling Center Kits": 0
    }
}

# Severity scaling factor for resources
RISK_SCALER = {
    "Low": 0.1,      # Base monitoring level
    "Medium": 0.6,   # Pre-positioning and local deployment
    "High": 1.3,     # Active deployment of regional resources
    "Severe": 2.8    # Full mobilization and state-wide reinforcement
}

def allocate_resources(disaster_type: str, risk_level: str, pop_density: float, vulnerability_index: float) -> dict:
    """
    Applies a multi-criteria optimization heuristic to calculate resources needed.
    
    Formula:
    R = Base_R * Risk_Scaler * Pop_Scaler * Vulnerability_Scaler
    
    Where:
    - Pop_Scaler scales logarithmically with population density: 
      1.0 + 0.4 * log10(pop_density / 100)
    - Vulnerability_Scaler scales linearly:
      vulnerability_index / 0.5 (where 0.5 is baseline)
    """
    if disaster_type not in BASE_RESOURCES:
        raise ValueError(f"Unknown disaster type: {disaster_type}")
        
    if risk_level not in RISK_SCALER:
        raise ValueError(f"Unknown risk level: {risk_level}")
        
    bases = BASE_RESOURCES[disaster_type]
    risk_factor = RISK_SCALER[risk_level]
    
    # Calculate population density scaler (logarithmic to prevent extreme explosions in dense cities)
    # Mumbai pop density is ~20,000, Gadchiroli is ~74. 
    # Reference density = 200 (typical rural/semi-urban district)
    density_ratio = max(0.1, pop_density / 200.0)
    pop_scaler = 1.0 + 0.4 * math.log10(density_ratio)
    pop_scaler = max(0.6, min(3.5, pop_scaler))  # Bound pop_scaler between 0.6 and 3.5
    
    # Calculate vulnerability scaler (linear relative to baseline of 0.7)
    vulnerability_scaler = vulnerability_index / 0.7
    vulnerability_scaler = max(0.5, min(1.5, vulnerability_scaler))
    
    allocations = {}
    for resource_name, base_val in bases.items():
        if base_val == 0:
            allocations[resource_name] = 0
            continue
            
        # Apply formula
        raw_allocated = base_val * risk_factor * pop_scaler * vulnerability_scaler
        
        # Round up to nearest integer to avoid fractional supplies
        allocated_val = math.ceil(raw_allocated)
        
        # Ensure a minimum allocation of 1 if risk is High or Severe and resource is applicable
        if risk_level in ["High", "Severe"] and allocated_val == 0:
            allocated_val = 1
            
        allocations[resource_name] = allocated_val
        
    return {
        "resource_allocation": allocations,
        "scalers": {
            "risk_scaler": round(risk_factor, 2),
            "pop_scaler": round(pop_scaler, 2),
            "vulnerability_scaler": round(vulnerability_scaler, 2),
            "combined_factor": round(risk_factor * pop_scaler * vulnerability_scaler, 2)
        }
    }

if __name__ == "__main__":
    # Test allocation
    test1 = allocate_resources("Flood", "Severe", 20980, 0.8) # Mumbai Suburban Flood
    print("Mumbai Suburban Severe Flood Allocation:")
    print(test1)
    
    print("\nGadchiroli High Flood Allocation:")
    test2 = allocate_resources("Flood", "High", 74, 0.85) # Gadchiroli Flood
    print(test2)
