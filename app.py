import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import traceback

st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

st.title("Used Car Price Predictor")
st.markdown("Estimate the fair market price of a used car based on its key attributes")

# Print to logs for debugging
print("Starting app...", flush=True)
print(f"Python version: {sys.version}", flush=True)

try:
    # Load model and preprocessing components
    @st.cache_resource
    def load_model():
        print("Loading model...", flush=True)
        return joblib.load('xgb_model_only.pkl')
    
    @st.cache_resource
    def load_scaler():
        print("Loading scaler...", flush=True)
        return joblib.load('numerical_scaler.pkl')
    
    @st.cache_resource
    def load_encoder():
        print("Loading encoder...", flush=True)
        return joblib.load('categorical_encoder.pkl')
    
    @st.cache_data
    def load_metadata():
        print("Loading metadata...", flush=True)
        with open('streamlit_metadata.json', 'r') as f:
            return json.load(f)
    
    @st.cache_data
    def load_preprocessing_info():
        print("Loading preprocessing info...", flush=True)
        with open('preprocessing_info.json', 'r') as f:
            return json.load(f)
    
    model = load_model()
    scaler = load_scaler()
    encoder = load_encoder()
    metadata = load_metadata()
    prep_info = load_preprocessing_info()
    print("All components loaded successfully", flush=True)
    
except Exception as e:
    st.error(f"Error loading model components: {str(e)}")
    st.code(traceback.format_exc())
    print(f"Error: {traceback.format_exc()}", flush=True)
    st.stop()

st.sidebar.header("Enter Car Details")

# Numerical inputs
model_year = st.sidebar.number_input("Model Year", min_value=1990, max_value=2025, value=2018, step=1)
milage = st.sidebar.number_input("Mileage (miles)", min_value=0, max_value=300000, value=50000, step=1000)
car_age = st.sidebar.number_input("Car Age (years)", min_value=0, max_value=35, value=5, step=1)
mileage_per_year = milage / (car_age + 1) if car_age >= 0 else milage

# Categorical inputs
brand_category = st.sidebar.selectbox("Brand", metadata['brand_categories'])
fuel_type = st.sidebar.selectbox("Fuel Type", metadata['fuel_types'])
transmission_simple = st.sidebar.selectbox("Transmission", metadata['transmission_types'])
accident = st.sidebar.selectbox("Accident History", metadata['accident_options'])
clean_title = st.sidebar.selectbox("Clean Title", metadata['clean_title_options'])
is_luxury = st.sidebar.selectbox("Luxury Brand?", ["No", "Yes"])
is_luxury_val = 1 if is_luxury == "Yes" else 0

if st.sidebar.button("Predict Price", type="primary"):
    try:
        # Prepare numerical features
        numerical_data = np.array([[model_year, milage, car_age, mileage_per_year]])
        numerical_scaled = scaler.transform(numerical_data)
        
        # Prepare categorical features
        categorical_data = pd.DataFrame([[transmission_simple, brand_category, fuel_type, accident, clean_title]],
                                        columns=prep_info['categorical_cols'])
        categorical_encoded = encoder.transform(categorical_data)
        
        # Binary feature
        binary_data = np.array([[is_luxury_val]])
        
        # Combine all features
        X_input = np.hstack([numerical_scaled, categorical_encoded, binary_data])
        
        # Predict (log scale)
        pred_log = model.predict(X_input)[0]
        pred_price = np.expm1(pred_log)
        
        st.markdown("## Estimated Price")
        st.markdown(f"### **${pred_price:,.0f}**")
        st.caption("This is an estimate based on historical data. Actual market price may vary.")
        
        with st.expander("View input details"):
            st.write(f"**Car age:** {car_age} years")
            st.write(f"**Mileage:** {milage:,} miles")
            st.write(f"**Mileage per year:** {mileage_per_year:.0f} miles/year")
            st.write(f"**Brand:** {brand_category}")
            st.write(f"**Clean title:** {clean_title}")
            st.write(f"**Accident:** {accident}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("*")
