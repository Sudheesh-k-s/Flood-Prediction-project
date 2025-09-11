
import streamlit as st
import joblib
import numpy as np

# Page setup
st.set_page_config(page_title="Flood Prediction", page_icon="🌊")
st.title("🌊 Flood Risk Prediction")
st.write("Enter parameters to predict flood risk:")

# Try to load the model
try:
    model = joblib.load('flood_prediction_model.pkl')
    st.success("Model loaded successfully!")
except:
    st.error("Model file not found. Please make sure 'flood_prediction_model.pkl' is in the same folder.")
    st.stop()

# Simple input form
st.subheader("Input Parameters")

# Numerical inputs
lat = st.number_input("Latitude", min_value=8.0, max_value=37.0, value=20.0)
lon = st.number_input("Longitude", min_value=68.0, max_value=97.0, value=78.0)
rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)
temp = st.number_input("Temperature (°C)", min_value=15.0, max_value=45.0, value=30.0)
humidity = st.number_input("Humidity (%)", min_value=20.0, max_value=100.0, value=60.0)
river = st.number_input("River Discharge (m³/s)", min_value=0.0, max_value=5000.0, value=2500.0)
water = st.number_input("Water Level (m)", min_value=0.0, max_value=10.0, value=5.0)
elev = st.number_input("Elevation (m)", min_value=0.0, max_value=9000.0, value=3000.0)
pop = st.number_input("Population Density", min_value=0.0, max_value=10000.0, value=5000.0)

# Categorical inputs
infra = st.radio("Infrastructure", ["No", "Yes"])
hist_flood = st.radio("Historical Floods", ["No", "Yes"])
land_cover = st.selectbox("Land Cover", ["Water Body", "Forest", "Agricultural", "Desert", "Urban"])
soil_type = st.selectbox("Soil Type", ["Clay", "Peat", "Loam", "Sandy", "Silt"])

# Convert categorical inputs to numerical
infra_code = 1 if infra == "Yes" else 0
hist_code = 1 if hist_flood == "Yes" else 0

# Simple encoding for land cover and soil type
land_mapping = {"Water Body": 0, "Forest": 1, "Agricultural": 2, "Desert": 3, "Urban": 4}
soil_mapping = {"Clay": 0, "Peat": 1, "Loam": 2, "Sandy": 3, "Silt": 4}

land_code = land_mapping[land_cover]
soil_code = soil_mapping[soil_type]

# Predict button
if st.button("Predict Flood Risk"):
    # Prepare features
    features = np.array([[lat, lon, rain, temp, humidity, river, water, elev, 
                         land_code, soil_code, pop, infra_code, hist_code]])
    
    # Make prediction
    try:
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Show result
        if prediction[0] == 1:
            st.error(f"🚨 Flood risk detected ({probability[0][1]:.1%} probability)")
            st.write("Take necessary precautions.")
        else:
            st.success(f"✅ Low flood risk ({probability[0][0]:.1%} probability)")
            st.write("No immediate danger, but stay alert.")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")