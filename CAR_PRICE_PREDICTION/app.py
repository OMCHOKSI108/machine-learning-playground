import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")

# # Load the model and preprocessors
# model = joblib.load('best_model_from_all.pkl')
# scaler = joblib.load('scaler.pkl')
# encoder = joblib.load('encoder.pkl')

model = joblib.load('/mount/src/machine-learning-playground/CAR_PRICE_PREDICTION/models/best_model_from_all.pkl')
scaler = joblib.load('/mount/src/machine-learning-playground/CAR_PRICE_PREDICTION/models/scaler.pkl')
encoder = joblib.load('/mount/src/machine-learning-playground/CAR_PRICE_PREDICTION/models/encoder.pkl')  

# Define features
num_features = ['enginesize', 'horsepower', 'curbweight', 'carwidth', 'wheelbase', 'citympg', 'doornumber', 'cylindernumber']
cat_features = ['brand', 'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem']

# Default values
defaults = {
    'enginesize': 130.0,
    'horsepower': 100.0,
    'curbweight': 2500.0,
    'carwidth': 66.0,
    'wheelbase': 99.0,
    'citympg': 25.0,
    'doornumber': 4,
    'cylindernumber': 4,
    'brand': 'toyota',
    'fueltype': 'gas',
    'aspiration': 'std',
    'carbody': 'sedan',
    'drivewheel': 'fwd',
    'enginelocation': 'front',
    'enginetype': 'ohc',
    'fuelsystem': 'mpfi'
}

# Categorical options (based on dataset)
cat_options = {
    'brand': ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi', 'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'],
    'fueltype': ['gas', 'diesel'],
    'aspiration': ['std', 'turbo'],
    'carbody': ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'],
    'drivewheel': ['rwd', 'fwd', '4wd'],
    'enginelocation': ['front', 'rear'],
    'enginetype': ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf'],
    'fuelsystem': ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi']
}

st.title("Car Price Prediction")
st.markdown("Enter the car features below to predict the price. Use the sliders and dropdowns for easy input.")

inputs = {}

# Create two columns for Numerical and Categorical sections
col_num, col_cat = st.columns(2)

with col_num:
    st.header("Numerical Features")
    with st.expander("Adjust Numerical Values", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inputs['enginesize'] = st.slider("Engine Size (cc)", min_value=61, max_value=326, value=int(defaults['enginesize']), step=1)
            inputs['horsepower'] = st.slider("Horsepower", min_value=48, max_value=288, value=int(defaults['horsepower']), step=1)
            inputs['curbweight'] = st.slider("Curb Weight (lbs)", min_value=1488, max_value=4066, value=int(defaults['curbweight']), step=1)
            inputs['doornumber'] = st.slider("Number of Doors", min_value=2, max_value=4, value=defaults['doornumber'], step=1)
        with col2:
            inputs['carwidth'] = st.slider("Car Width (in)", min_value=60.3, max_value=72.3, value=defaults['carwidth'], step=0.1)
            inputs['wheelbase'] = st.slider("Wheelbase (in)", min_value=86.6, max_value=120.9, value=defaults['wheelbase'], step=0.1)
            inputs['citympg'] = st.slider("City MPG", min_value=13, max_value=49, value=int(defaults['citympg']), step=1)
            inputs['cylindernumber'] = st.slider("Number of Cylinders", min_value=2, max_value=12, value=defaults['cylindernumber'], step=1)

with col_cat:
    st.header("Categorical Features")
    with st.expander("Select Categorical Options", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inputs['brand'] = st.selectbox("Brand", options=cat_options['brand'], index=cat_options['brand'].index(defaults['brand']))
            inputs['fueltype'] = st.selectbox("Fuel Type", options=cat_options['fueltype'], index=cat_options['fueltype'].index(defaults['fueltype']))
            inputs['aspiration'] = st.selectbox("Aspiration", options=cat_options['aspiration'], index=cat_options['aspiration'].index(defaults['aspiration']))
            inputs['carbody'] = st.selectbox("Car Body", options=cat_options['carbody'], index=cat_options['carbody'].index(defaults['carbody']))
        with col2:
            inputs['drivewheel'] = st.selectbox("Drive Wheel", options=cat_options['drivewheel'], index=cat_options['drivewheel'].index(defaults['drivewheel']))
            inputs['enginelocation'] = st.selectbox("Engine Location", options=cat_options['enginelocation'], index=cat_options['enginelocation'].index(defaults['enginelocation']))
            inputs['enginetype'] = st.selectbox("Engine Type", options=cat_options['enginetype'], index=cat_options['enginetype'].index(defaults['enginetype']))
            inputs['fuelsystem'] = st.selectbox("Fuel System", options=cat_options['fuelsystem'], index=cat_options['fuelsystem'].index(defaults['fuelsystem']))

# Predict button
st.header("Prediction")
if st.button("Predict Price", type="primary"):
    # Create input DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Predict (model handles preprocessing)
    pred_log = model.predict(input_df)
    predicted_price = np.expm1(pred_log)[0]
    
    st.success(f"Predicted Car Price: ₹‎{predicted_price:,.2f}")