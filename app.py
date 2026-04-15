
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

st.title("Used Car Price Predictor")
st.markdown("Estimate the fair market price of a used car based on its key attributes")

# Load model and metadata
@st.cache_resource
def load_model():
    return joblib.load('used_car_predictor.pkl')

@st.cache_data
def load_metadata():
    with open('streamlit_metadata.json', 'r') as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

st.sidebar.header("Enter Car Details")

# Numerical inputs
model_year = st.sidebar.number_input("Model Year", min_value=1990, max_value=2025, value=2018, step=1)
milage = st.sidebar.number_input("Mileage (miles)", min_value=0, max_value=300000, value=50000, step=1000)
car_age = st.sidebar.number_input("Car Age (years)", min_value=0, max_value=35, value=5, step=1)

# Derived feature (mileage per year) - auto-calculated
mileage_per_year = milage / (car_age + 1) if car_age >= 0 else milage

# Categorical inputs
brand_category = st.sidebar.selectbox("Brand", metadata['brand_categories'])
fuel_type = st.sidebar.selectbox("Fuel Type", metadata['fuel_types'])
transmission_simple = st.sidebar.selectbox("Transmission", metadata['transmission_types'])
accident = st.sidebar.selectbox("Accident History", metadata['accident_options'])
clean_title = st.sidebar.selectbox("Clean Title", metadata['clean_title_options'])
is_luxury = st.sidebar.selectbox("Luxury Brand?", ["No", "Yes"])

# Convert luxury to 0/1
is_luxury_val = 1 if is_luxury == "Yes" else 0

if st.sidebar.button("Predict Price", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame([{
        'model_year': model_year,
        'milage': milage,
        'car_age': car_age,
        'is_luxury': is_luxury_val,
        'mileage_per_year': mileage_per_year,
        'transmission_simple': transmission_simple,
        'brand_category': brand_category,
        'fuel_type': fuel_type,
        'accident': accident,
        'clean_title': clean_title
    }])
    
    # Predict (log scale)
    pred_log = model.predict(input_data)[0]
    pred_price = np.expm1(pred_log)
    
    # Display
    st.markdown("## Estimated Price")
    st.markdown(f"### **${pred_price:,.0f}**")
    
    # Optional: show confidence note
    st.caption("This is an estimate based on historical data. Actual market price may vary due to condition, location, and demand.")
    
    # Show input summary
    with st.expander("View input details"):
        st.write(f"**Car age:** {car_age} years")
        st.write(f"**Mileage:** {milage:,} miles")
        st.write(f"**Mileage per year:** {mileage_per_year:.0f} miles/year")
        st.write(f"**Brand:** {brand_category}")
        st.write(f"**Clean title:** {clean_title}")
        st.write(f"**Accident:** {accident}")

st.markdown("---")
st.markdown("*Model trained on used car dataset. For reference only.*")
