import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import math
import requests
from fetch_real_nasa_data import DISTRICT_COORDS

# Helper function to fetch live weather from Open-Meteo
def fetch_live_weather(district_name: str) -> dict:
    if district_name not in DISTRICT_COORDS:
        return None
    coords = DISTRICT_COORDS[district_name]
    lat, lon = coords["lat"], coords["lon"]
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&timezone=auto"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            res_data = response.json()
            current = res_data["current"]
            return {
                "temperature": float(current["temperature_2m"]),
                "humidity": float(current["relative_humidity_2m"]),
                "wind_speed": float(current["wind_speed_10m"]),
                "rainfall": float(current["precipitation"])
            }
    except Exception as e:
        pass
    return None

from data_loader import DISTRICTS_METADATA
from resource_allocator import allocate_resources
import sqlite3
from datetime import datetime

# Persistent database helpers
DB_FILE = "disaster_helpline.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS help_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT,
            district TEXT,
            location_details TEXT,
            emergency_type TEXT,
            urgency TEXT,
            status TEXT DEFAULT 'Pending',
            eta_minutes INTEGER,
            response_notes TEXT,
            submitted_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def add_help_request(name, phone, district, location, em_type, urgency):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO help_requests (name, phone, district, location_details, emergency_type, urgency, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name, phone, district, location, em_type, urgency, submitted_at))
    conn.commit()
    conn.close()

def get_request_by_phone(phone):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM help_requests WHERE phone = ? ORDER BY submitted_at DESC", (phone,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def get_all_requests():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM help_requests ORDER BY submitted_at DESC")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def update_request_status(req_id, status, eta, notes):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        UPDATE help_requests
        SET status = ?, eta_minutes = ?, response_notes = ?
        WHERE id = ?
    """, (status, eta, notes, req_id))
    conn.commit()
    conn.close()

# Page configuration
st.set_page_config(
    page_title="AI Disaster Management & Resource Allocation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 5px;
    }
    .risk-low {
        background-color: #15803d;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .risk-medium {
        background-color: #a16207;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .risk-high {
        background-color: #c2410c;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .risk-severe {
        background-color: #991b1b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load ML Bundle
@st.cache_resource
def load_model_bundle():
    model_path = os.path.join(os.path.dirname(__file__), "disaster_risk_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"Trained model not found at {model_path}. Run train_model.py first.")
        st.stop()
    return joblib.load(model_path)

bundle = load_model_bundle()
model = bundle["model"]
district_encoder = bundle["district_encoder"]
risk_reverse_mapping = bundle["risk_reverse_mapping"]
feature_cols = bundle["feature_cols"]
metrics = bundle["metrics"]

districts = sorted(list(DISTRICTS_METADATA.keys()))
disasters = ["Flood", "Cyclone", "Landslide", "Heatwave", "Drought"]

SOP_TEMPLATES = {
    "Flood": [
        "⚠️ Move immediately to higher ground or designated shelter buildings.",
        "🔌 Disconnect electrical appliances and main switches. Do not touch electrical equipment if wet.",
        "🚗 Do not attempt to drive or walk through moving flood waters.",
        "🚰 Drink only boiled or bottled water to prevent waterborne contamination.",
        "📻 Monitor local news and radio broadcasts for rising river level updates."
    ],
    "Cyclone": [
        "🏠 Stay indoors, away from windows, and shut all doors securely.",
        "🎒 Keep an emergency preparedness kit ready (dry food, water, flashlight, first aid, medicines).",
        "🌳 Steer clear of old structures, metal sheets, and tall trees that could collapse.",
        "📻 Listen to official bulletins for land-fall updates.",
        "🌊 If in a coastal low-lying area, evacuate inland immediately to safety shelters."
    ],
    "Landslide": [
        "🏃 Evacuate immediately if you hear unusual sounds (cracking trees, boulders knocking).",
        "🏠 Stay alert during heavy rainstorms, especially if living on a sloped terrain.",
        "🚧 Stay away from active landslide paths and avoid traveling in mountainous ghats.",
        "🛌 If trapped indoors, curl into a tight ball under sturdy furniture and protect your head.",
        "📢 Report any ground cracks or sudden soil movements to local authorities."
    ],
    "Heatwave": [
        "🚰 Drink ample water regularly, even if not thirsty. Avoid alcohol and caffeinated drinks.",
        "🧴 Limit direct outdoor sun exposure, particularly between 12:00 PM and 3:00 PM.",
        "👕 Wear light-colored, loose, breathable cotton clothes.",
        "🐄 Keep domestic cattle and pets under shade and ensure they have access to water.",
        "🏥 Watch for signs of heat exhaustion: dizziness, nausea, headaches, or high body temperature."
    ],
    "Drought": [
        "🚰 Conserve water strictly; prioritize water for drinking, cooking, and sanitation.",
        "🌾 Implement drip irrigation and mulching to sustain agricultural crops with minimal water.",
        "💧 Harvest rainwater and recycle domestic greywater.",
        "🚜 Register with local agricultural officers for drought relief support.",
        "🐄 Move cattle to state-sponsored fodder camps if water and feed are depleted."
    ]
}

def generate_helpline(district_name: str) -> str:
    if "Mumbai" in district_name:
        code = "022"
    elif district_name == "Pune":
        code = "020"
    elif district_name == "Nagpur":
        code = "0712"
    elif district_name == "Nashik":
        code = "0253"
    elif district_name == "Aurangabad":
        code = "0240"
    else:
        code = "0" + str(200 + (hash(district_name) % 700))
    number = str(220000 + (hash(district_name) % 9999))
    return f"{code}-{number}"

# Header Banner
st.title("🛡️ AI Disaster Management & Response System")
st.markdown("##### State Disaster Relief Operations and Predictive Modeling Portal (Maharashtra, India)")
st.write("---")

# Sidebar View Selector
st.sidebar.subheader("🔒 Access Portal Selection")
app_mode = st.sidebar.radio("View Mode", ["👤 Citizen Portal", "🛡️ Admin Operations Dashboard"])

# ==================== CITIZEN PORTAL ====================
if app_mode == "👤 Citizen Portal":
    st.subheader("🆘 Citizen Emergency Assistance & Relief Registration")
    st.write("If you are in distress or need emergency help during a disaster, register your request below. You can also track relief status and view helpline numbers.")
    
    tab_register, tab_track, tab_helplines = st.tabs([
        "🚨 Register Emergency / Raise Help Request",
        "🔍 Track Relief Status",
        "📞 Emergency Helplines Directory"
    ])
    
    with tab_register:
        st.write("### Raise Emergency Relief Call")
        with st.form("citizen_help_form", clear_on_submit=True):
            cit_name = st.text_input("Your Full Name", placeholder="e.g. Rahul Sharma")
            cit_phone = st.text_input("Mobile Phone Number (10 digits)", placeholder="e.g. 9876543210")
            cit_district = st.selectbox("Select Your District", districts, key="citizen_district")
            cit_location = st.text_area("Specific Landmark / Location Details", placeholder="e.g. Lane 3, near Ram Temple, ground floor flooded")
            
            cit_em_type = st.selectbox("Relief Needed", [
                "Medical Support (First Aid / Ambulance)",
                "Evacuation / Search & Rescue (Boat / Personnel Required)",
                "Essential Supplies (Food / Drinking Water)",
                "Emergency Shelter / Tents",
                "Fire Outbreak / Safety Hazard",
                "Other / General Assistance"
            ])
            
            cit_urgency = st.selectbox("Urgency Level", [
                "Immediate / Life-Threatening",
                "Critical / High Priority",
                "Moderate / Needs Assistance"
            ])
            
            submitted = st.form_submit_button("🚨 Submit Help Request")
            
            if submitted:
                if not cit_name or not cit_phone or not cit_location:
                    st.error("All fields (Name, Phone, and Location Details) are required.")
                elif not cit_phone.isdigit() or len(cit_phone) < 10:
                    st.error("Please enter a valid 10-digit mobile number.")
                else:
                    add_help_request(cit_name, cit_phone, cit_district, cit_location, cit_em_type, cit_urgency)
                    st.success(f"Emergency request registered successfully for {cit_name}! The administrative relief team has been alerted.")
                    st.info("You can track your request status in the 'Track Relief Status' tab using your mobile number.")

    with tab_track:
        st.write("### Track Relief Dispatch Status")
        search_phone = st.text_input("Enter your registered mobile number", placeholder="e.g. 9876543210")
        
        if search_phone:
            if not search_phone.isdigit():
                st.error("Please enter digits only.")
            else:
                user_requests = get_request_by_phone(search_phone)
                if not user_requests:
                    st.warning("No relief request found for this phone number.")
                else:
                    for req in user_requests:
                        st.write("---")
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            st.markdown(f"**Request ID:** #{req['id']}")
                            st.markdown(f"**Disaster Context / Need:** {req['emergency_type']}")
                            st.markdown(f"**Urgency:** {req['urgency']}")
                            st.markdown(f"**Submitted At:** {req['submitted_at']}")
                            st.markdown(f"📍 **District:** {req['district']} | **Location:** {req['location_details']}")
                        with col_r2:
                            if req['status'] == 'Pending':
                                st.markdown('<div style="background-color: #a16207; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;">🟡 Relief Request Pending Response</div>', unsafe_allow_html=True)
                                st.write("Relief desk is assessing the request and assigning response units.")
                            else:
                                st.markdown('<div style="background-color: #15803d; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;">🟢 Relief Dispatched / Action Taken</div>', unsafe_allow_html=True)
                                st.success(f"**Dispatch Status:** Active Dispatch")
                                if req['eta_minutes']:
                                    st.info(f"⏱️ **Estimated Arrival Time (ETA):** {req['eta_minutes']} minutes")
                                else:
                                    st.info("⏱️ **Estimated Arrival Time (ETA):** Dispatch unit has departed, arriving shortly.")
                                if req['response_notes']:
                                    st.write(f"**Relief Commander Notes:** *{req['response_notes']}*")

    with tab_helplines:
        st.header("📞 Maharashtra Emergency Assistance Directory")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.subheader("🏢 Central State Control Rooms")
            st.markdown("""
            - **Maharashtra State Emergency Operations Center (SEOC):** `+91-22-22027990` / `+91-22-22854746`
            - **State Disaster Response Force (SDRF) HQ:** `+91-20-25667300`
            - **National Emergency Toll-Free Number:** `1070` or `108` (Medical Emergencies)
            - **State Relief and Rehabilitation Department:** `1077` (District-Specific Routing)
            """)
        with col_h2:
            st.subheader("📍 Local Municipal Emergency Rooms")
            st.markdown("""
            - **Local EOC Routing:** Dial `1077` from local landlines to route directly to your local district headquarters.
            - **Local Police Control Room:** `100` | **Fire Brigade:** `101`
            - **Ambulance Service:** `102` / `108`
            """)

# ==================== ADMIN OPERATIONS DASHBOARD ====================
elif app_mode == "🛡️ Admin Operations Dashboard":
    st.subheader("🛡️ Administrative Command & Control Dashboard")
    st.write("Emergency response command center. Review distress queues, execute predictive models, and optimize relief supplies dispatch.")
    
    admin_pass = st.text_input("Enter Admin Access Code", type="password", placeholder="Enter authorization key (use 'admin')")
    
    if admin_pass == "admin" or admin_pass == "relief123":
        st.success("Authorized Access Granted.")
        
        tab_queue, tab_pred, tab_diag = st.tabs([
            "📋 Citizen Requests & Relief Dispatch Queue",
            "🔮 ML Risk Predictor & Resource Allocator",
            "📊 ML Model Performance & Math Diagnostics"
        ])
        
        with tab_queue:
            st.subheader("📋 distress calls queue & Situational status")
            all_reqs = get_all_requests()
            pending_count = sum(1 for r in all_reqs if r["status"] == "Pending")
            dispatch_count = sum(1 for r in all_reqs if r["status"] == "Dispatched")
            
            col_ad_st1, col_ad_st2, col_ad_st3 = st.columns(3)
            col_ad_st1.metric("Total distress Calls", len(all_reqs))
            col_ad_st2.metric("Pending Response", pending_count)
            col_ad_st3.metric("Relief Dispatched", dispatch_count)
            
            if not all_reqs:
                st.info("No distress calls registered in SQLite database.")
            else:
                for r in all_reqs:
                    st.write("---")
                    col_ad_q1, col_ad_q2 = st.columns([2, 1])
                    with col_ad_q1:
                        urgency_color = "red" if r["urgency"] == "Immediate / Life-Threatening" else ("orange" if r["urgency"] == "Critical / High Priority" else "#fbbf24")
                        st.markdown(f"##### Request #{r['id']} - {r['name']} ({r['phone']})")
                        st.markdown(f"📍 **District:** {r['district']} | **Location Details:** {r['location_details']}")
                        st.markdown(f"🚨 **Emergency Type:** {r['emergency_type']} | <span style='color:{urgency_color}; font-weight:bold;'>Urgency: {r['urgency']}</span>", unsafe_allow_html=True)
                        st.write(f"🕒 **Submitted:** {r['submitted_at']} | **Current Status:** `{r['status']}`")
                        if r['status'] == 'Dispatched':
                            st.write(f"🟢 **Dispatch Details:** ETA: {r['eta_minutes']} min | *Notes: {r['response_notes']}*")
                    with col_ad_q2:
                        if r['status'] == 'Pending':
                            with st.expander("⚡ Dispatch Relief Unit"):
                                with st.form(f"dispatch_form_ad_{r['id']}"):
                                    eta_val = st.number_input("Estimated Time of Arrival (minutes)", min_value=5, max_value=240, value=30, step=5)
                                    op_notes = st.text_area("Relief Team Operational Notes", placeholder="e.g. 2 search rafts and medical kits dispatched from closest local center.")
                                    dispatch_btn = st.form_submit_button("🚀 Confirm Dispatch & Notify Citizen")
                                    if dispatch_btn:
                                        update_request_status(r['id'], "Dispatched", int(eta_val), op_notes)
                                        st.success("Relief unit successfully dispatched! Citizen page updated.")
                                        st.rerun()
                        else:
                            st.info("Action Completed ✅")
                            
        with tab_pred:
            st.subheader("🔮 Predictive Risk Analysis & Relief Estimator")
            col_in_meta, col_in_metrics = st.columns([1, 2])
            
            with col_in_meta:
                st.subheader("📍 Location & Emergency Context")
                selected_district = st.selectbox("Select Maharashtra District", districts)
                selected_disaster = st.selectbox("Select Disaster Hazard Type", disasters)
                meta = DISTRICTS_METADATA[selected_district]
                
                st.markdown(f"""
                <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-top: 10px;">
                    <p style="margin-bottom: 5px;"><b>Region:</b> {meta['region']}</p>
                    <p style="margin-bottom: 5px;"><b>Population Density:</b> {meta['pop_density']:,} / km²</p>
                    <p style="margin-bottom: 5px;"><b>Average Slope:</b> {meta['slope']}%</p>
                    <p style="margin-bottom: 5px;"><b>Coastal District:</b> {"Yes" if meta['coastal'] == 1 else "No"}</p>
                    <p style="margin-bottom: 0px;"><b>Infrastructure Vulnerability Index:</b> {meta['vulnerability']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col_in_metrics:
                st.subheader("🌡️ Environmental Parameter Inputs")
                use_live_weather = st.checkbox("📥 Fetch Live Weather Data (via Open-Meteo API)", value=False)
                
                live_data = None
                if use_live_weather:
                    with st.spinner(f"Querying Open-Meteo API for {selected_district}..."):
                        live_data = fetch_live_weather(selected_district)
                        if live_data:
                            st.success(f"Successfully loaded live weather data for {selected_district}!")
                        else:
                            st.error("Could not reach API. Falling back to historical defaults.")
                            
                if live_data:
                    def_rain = live_data["rainfall"]
                    def_wind = live_data["wind_speed"]
                    def_temp = live_data["temperature"]
                    def_hum = live_data["humidity"]
                    def_sm = min(1.0, max(0.05, 0.40 + (live_data["rainfall"] / 100.0)))
                    def_riv = round(-1.5 + (live_data["rainfall"] / 40.0), 2)
                else:
                    if selected_disaster == "Flood":
                        def_rain, def_wind, def_temp, def_hum, def_sm, def_riv = 190.0, 32.0, 26.5, 92.0, 0.85, 2.40
                    elif selected_disaster == "Cyclone":
                        def_rain, def_wind, def_temp, def_hum, def_sm, def_riv = 230.0, 135.0 if meta["coastal"] == 1 else 85.0, 25.0, 95.0, 0.80, 1.60
                    elif selected_disaster == "Landslide":
                        def_rain, def_wind, def_temp, def_hum, def_sm, def_riv = 165.0, 22.0, 24.5, 88.0, 0.90, -0.50
                    elif selected_disaster == "Heatwave":
                        def_rain, def_wind, def_temp, def_hum, def_sm, def_riv = 0.0, 14.0, 44.5 if meta["region"] == "Vidarbha" else 42.0, 18.0, 0.14, -1.80
                    elif selected_disaster == "Drought":
                        def_rain, def_wind, def_temp, def_hum, def_sm, def_riv = 2.5, 10.0, 38.5, 28.0, 0.07, -2.10
                    else:
                        def_rain, def_wind, def_temp, def_hum, def_sm, def_riv = 50.0, 15.0, 30.0, 50.0, 0.40, 0.0
                        
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    rainfall = st.slider("Rainfall in last 24h (mm)", 0.0, 500.0, def_rain, step=1.0)
                    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 250.0, def_wind, step=1.0)
                    temperature = st.slider("Temperature (°C)", 10.0, 55.0, def_temp, step=0.5)
                with col_m2:
                    humidity = st.slider("Humidity (%)", 5.0, 100.0, def_hum, step=1.0)
                    soil_moisture = st.slider("Soil Moisture Index (SMI)", 0.0, 1.0, def_sm, step=0.01)
                    river_level = st.slider("River Level relative to danger mark (m)", -5.0, 10.0, def_riv, step=0.1)
                    
            st.write("---")
            predict_btn = st.button("🔮 Run Risk Prediction & Resource Allocation Model", type="primary", use_container_width=True)
            
            if predict_btn:
                district_enc = district_encoder.transform([selected_district])[0]
                feature_data = {
                    "district_enc": district_enc,
                    "pop_density": meta["pop_density"],
                    "slope": meta["slope"],
                    "coastal": meta["coastal"],
                    "vulnerability_index": meta["vulnerability"],
                    "rainfall": rainfall,
                    "wind_speed": wind_speed,
                    "temperature": temperature,
                    "humidity": humidity,
                    "soil_moisture": soil_moisture,
                    "river_level": river_level
                }
                X_infer = pd.DataFrame([feature_data])[feature_cols]
                
                risk_class_idx = model.predict(X_infer)[0]
                risk_prob = model.predict_proba(X_infer)[0]
                risk_level = risk_reverse_mapping[risk_class_idx]
                
                alloc_results = allocate_resources(selected_disaster, risk_level, meta["pop_density"], meta["vulnerability"])
                resources = alloc_results["resource_allocation"]
                scalers = alloc_results["scalers"]
                
                col_out_risk, col_out_alloc = st.columns([1, 2])
                with col_out_risk:
                    st.subheader("🚨 Risk Prediction Result")
                    if risk_level == "Low":
                        st.markdown("<div class='risk-low'>🟢 LOW RISK</div>", unsafe_allow_html=True)
                    elif risk_level == "Medium":
                        st.markdown("<div class='risk-medium'>🟡 MEDIUM RISK</div>", unsafe_allow_html=True)
                    elif risk_level == "High":
                        st.markdown("<div class='risk-high'>🟠 HIGH RISK</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='risk-severe'>🔴 SEVERE RISK</div>", unsafe_allow_html=True)
                        
                    st.write("")
                    prob_df = pd.DataFrame({
                        "Risk Level": [risk_reverse_mapping[i] for i in range(4)],
                        "Confidence Score": [f"{p*100:.1f}%" for p in risk_prob]
                    })
                    st.table(prob_df)
                    st.markdown(f"""
                    **Allocation Scaling Summary:**
                    - Base Severity Multiplier: `{scalers['risk_scaler']}x`
                    - Population Scale Factor: `{scalers['pop_scaler']}x`
                    - Vulnerability Adjustment: `{scalers['vulnerability_scaler']}x`
                    - **Combined Intensity Factor:** `{scalers['combined_factor']}x`
                    """)
                    
                with col_out_alloc:
                    st.subheader("📦 Optimized Emergency Resource Allocation")
                    st.caption("Calculated via Multi-Criteria Optimization (log-scaled on district vulnerability and population)")
                    
                    r_cols = st.columns(3)
                    active_resources = {k: v for k, v in resources.items() if v > 0}
                    res_items = list(active_resources.items())
                    
                    for idx, (res_name, res_qty) in enumerate(res_items):
                        col_target = r_cols[idx % 3]
                        icon = "📦"
                        if "Boat" in res_name: icon = "🚣"
                        elif "Medical" in res_name: icon = "🏥"
                        elif "Food" in res_name: icon = "🍞"
                        elif "Personnel" in res_name: icon = "👨‍🚒"
                        elif "Tent" in res_name: icon = "⛺"
                        elif "Tanker" in res_name: icon = "🚛"
                        elif "Cooling" in res_name: icon = "❄️"
                        
                        col_target.markdown(f"""
                        <div class='metric-card'>
                            <div style='font-size: 30px;'>{icon}</div>
                            <div class='metric-val'>{res_qty:,}</div>
                            <div class='metric-lbl'>{res_name}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    if not active_resources:
                        st.info("No active emergency resources required for Low risk levels of this hazard.")
                        
        with tab_diag:
            st.header("📊 Machine Learning Performance & Mathematics")
            col_d1, col_d2 = st.columns([1, 1])
            
            with col_d1:
                st.subheader("📈 Model Training Metrics")
                st.write(f"**Random Forest Test Accuracy:** `{metrics['test_accuracy']*100:.2f}%`")
                st.write(f"**5-Fold Cross-Validation Accuracy:** `{metrics['cv_mean']*100:.2f}% (+/- {metrics['cv_std']*100:.2f}%)`")
                
                st.write("**Classification Report:**")
                cls_rep = metrics["classification_report"]
                report_df = pd.DataFrame(cls_rep).transpose()
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)
                
            with col_d2:
                st.subheader("🏆 Feature Importance")
                imp_path = os.path.join(os.path.dirname(__file__), "feature_importance.csv")
                if os.path.exists(imp_path):
                    imp_df = pd.read_csv(imp_path)
                    imp_df["feature"] = imp_df["feature"].str.replace("_enc", "")
                    st.bar_chart(data=imp_df, x="feature", y="importance", color="#0ea5e9", use_container_width=True)
                else:
                    st.write("Feature importances file not found.")
                    
            st.write("---")
            st.subheader("📝 Mathematical Optimization Model (IEEE Formulation)")
            st.markdown(r"""
            The resource allocation engine is modeled as a multi-criteria optimization heuristic, defined as:
            
            $$R_{allocated} = \left\lceil \text{Base}_{R, d} \times S_{\text{risk}} \times P_{\text{pop}} \times V_{\text{vuln}} \right\rceil$$
            
            Where:
            - $\text{Base}_{R, d}$ is the baseline demand vector for resource $R$ and disaster $d$.
            - $S_{\text{risk}} \in \{0.1, 0.6, 1.3, 2.8\}$ is the ordinal risk scaling coefficient derived from the Random Forest model.
            - $P_{\text{pop}} = \max\left(0.6, \min\left(3.5, 1.0 + 0.4 \log_{10}\left(\frac{\text{Density}}{200}\right)\right)\right)$ represents the logarithmic population density scaling factor.
            - $V_{\text{vuln}} = \max\left(0.5, \min\left(1.5, \frac{\text{Index}}{0.7}\right)\right)$ is the infrastructure vulnerability scaling coefficient.
            
            This formulation ensures bounded resource allocation across extreme geographical density variants (e.g. comparing Mumbai Suburban to Gadchiroli) without creating computational overflow.
            """)
            st.info("📑 A detailed IEEE research draft is available in your workspace as `ieee_research_paper.md`.")
            
    elif admin_pass:
        st.error("Invalid credentials. Please enter 'admin' in the access code field.")
    else:
        st.info("🔒 Provide the Admin Access Code in the sidebar or command panel to view operational details.")
