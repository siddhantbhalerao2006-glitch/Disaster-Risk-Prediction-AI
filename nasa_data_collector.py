import requests
import pandas as pd
import time
import os

# Coordinates (Latitude, Longitude) for major districts of Maharashtra
MAHARASHTRA_COORDINATES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Nashik": {"lat": 19.9975, "lon": 73.7898},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433},
    "Ratnagiri": {"lat": 16.9902, "lon": 73.3120},
    "Beed": {"lat": 18.9890, "lon": 75.7601},
    "Gadchiroli": {"lat": 20.1005, "lon": 80.0001},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433},
    "Amravati": {"lat": 20.9320, "lon": 77.7523},
}

# NASA POWER API daily endpoint
# Parameters requested:
# - PRECTOTCORR: Precipitation Corrected (Rainfall in mm/day)
# - WS2M: Wind Speed at 2 Meters (km/h conversion needed: m/s * 3.6)
# - T2M: Temperature at 2 Meters (°C)
# - RH2M: Relative Humidity at 2 Meters (%)
# - GWETTOP: Top Profile Soil Moisture (Index 0 to 1)
NASA_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def fetch_nasa_data(district: str, lat: float, lon: float, start_date: str, end_date: str):
    """
    Fetches daily meteorological data from NASA POWER API for a given location and date range.
    Dates format: YYYYMMDD (e.g., '20250601' to '20250930' for monsoon 2025).
    """
    params = {
        "parameters": "PRECTOTCORR,WS2M,T2M,RH2M,GWETTOP",
        "community": "ag",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    print(f"Fetching daily weather for {district} ({lat}, {lon}) from {start_date} to {end_date}...")
    try:
        response = requests.get(NASA_API_URL, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            properties = data["properties"]["parameter"]
            
            # Reconstruct daily records
            dates = list(properties["T2M"].keys())
            records = []
            for date in dates:
                # Convert wind speed from m/s to km/h
                wind_speed_kmh = properties["WS2M"][date] * 3.6 if properties["WS2M"][date] != -999 else -999
                
                records.append({
                    "date": date,
                    "district": district,
                    "rainfall": properties["PRECTOTCORR"][date],
                    "wind_speed": round(wind_speed_kmh, 1),
                    "temperature": properties["T2M"][date],
                    "humidity": properties["RH2M"][date],
                    "soil_moisture": properties["GWETTOP"][date]
                })
            return pd.DataFrame(records)
        else:
            print(f"Error: Received status code {response.status_code} from API.")
            return None
    except Exception as e:
        print(f"Network error fetching data for {district}: {str(e)}")
        return None

def collect_historical_dataset(start_date: str = "20250601", end_date: str = "20251031"):
    """
    Collects and merges meteorological data for multiple districts.
    """
    all_data = []
    
    for district, coords in MAHARASHTRA_COORDINATES.items():
        df_district = fetch_nasa_data(
            district, coords["lat"], coords["lon"], start_date, end_date
        )
        if df_district is not None:
            all_data.append(df_district)
            # Sleep briefly to respect NASA API rate limits
            time.sleep(1.5)
            
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        # Filter out missing sensor values (often represented as -999)
        merged_df = merged_df[merged_df["temperature"] != -999]
        
        out_path = "nasa_raw_meteorology.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"\nSuccessfully collected daily parameters! Saved to: {out_path}")
        print(merged_df.head(10))
    else:
        print("Failed to collect any data.")

if __name__ == "__main__":
    # Test script: Fetch data for Monsoon 2025 (June 1 to October 31, 2025)
    # Format: YYYYMMDD
    collect_historical_dataset(start_date="20250601", end_date="20251031")
