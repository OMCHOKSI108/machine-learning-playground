import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder

# Function to apply feature engineering (from notebook)
def Feature_E(df):
    # Feature 1: BMI (Body Mass Index)
    df['BMI'] = df['Weight'] / (df['Height'] / 100)**2

    # Feature 2: Number of meals per day
    df['Meals_Per_Day'] = df['FCVC'] + df['NCP']

    # Feature 3: Total physical activity score
    df['Total_Activity_Score'] = df['FAF'] * df['TUE']

    # Feature 5: Age category (e.g., young, adult, elderly)
    df['Age_Category'] = pd.cut(df['Age'], bins=[0, 18, 60, float('inf')], labels=['Young', 'Adult', 'Elderly'])

    # Feature 6: Water intake per kg of body weight
    df['Water_Intake_Per_Kg'] = df['CH2O'] / df['Weight']

    return df

# Load the saved model
model = joblib.load('streamlit_assets/best_model_pipeline.pkl')
unique_classes = joblib.load('streamlit_assets/unique_classes.pkl')

# Recreate and fit transformers on training data
tr_d = pd.read_csv('train.csv')
O_D = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
tr_d = pd.concat([tr_d, O_D], ignore_index=True)
tr_d.drop_duplicates(inplace=True)
tr_d = Feature_E(tr_d)

numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Meals_Per_Day', 'Total_Activity_Score', 'Water_Intake_Per_Kg']
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'Age_Category']

scaler = QuantileTransformer(output_distribution='normal', random_state=42)
scaler.fit(tr_d[numerical_cols])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(tr_d[categorical_cols])

# Streamlit app
st.title("Obesity Level Prediction")

st.markdown("Enter the details below to predict the obesity level.")

# Numerical inputs (above)
st.header("Numerical Inputs")
age = st.number_input("Age", min_value=0, max_value=100, value=25, step=1)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
fcvc = st.number_input("FCVC (Frequency of vegetable consumption)", min_value=1, max_value=3, value=2, step=1)
ncp = st.number_input("NCP (Number of main meals)", min_value=1, max_value=4, value=3, step=1)
ch2o = st.number_input("CH2O (Daily water intake)", min_value=1, max_value=3, value=2, step=1)
faf = st.number_input("FAF (Physical activity frequency)", min_value=0, max_value=3, value=1, step=1)
tue = st.number_input("TUE (Time using technology)", min_value=0, max_value=2, value=1, step=1)

# Categorical inputs (below)
st.header("Categorical Inputs")
gender = st.selectbox("Gender", ["Male", "Female"], index=0)
family_history = st.selectbox("Family history with overweight", ["yes", "no"], index=0)
favc = st.selectbox("FAVC (Frequent high caloric food)", ["yes", "no"], index=0)
caec = st.selectbox("CAEC (Food between meals)", ["Sometimes", "Frequently", "Always", "no"], index=0)
smoke = st.selectbox("Smoke", ["no", "yes"], index=0)
scc = st.selectbox("SCC (Monitor calories)", ["no", "yes"], index=0)
calc = st.selectbox("CALC (Alcohol consumption)", ["Sometimes", "no", "Frequently", "Always"], index=0)
mtrans = st.selectbox("MTRANS (Transportation)", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"], index=0)

# Prediction button
if st.button("Predict Obesity Level"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })

    # Apply feature engineering
    input_data = Feature_E(input_data)

    # Define columns
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Meals_Per_Day', 'Total_Activity_Score', 'Water_Intake_Per_Kg']
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'Age_Category']

    # Scale numerical
    scaled_numerical = scaler.transform(input_data[numerical_cols])

    # Encode categorical
    encoded_categorical = encoder.transform(input_data[categorical_cols])

    # Combine
    input_final_array = np.hstack([scaled_numerical, encoded_categorical])

    # Predict
    prediction = model.predict(input_final_array)
    predicted_class = prediction[0]

    # Debug outputs
    st.write(f"Scaled numerical shape: {scaled_numerical.shape}")
    st.write(f"Encoded categorical shape: {encoded_categorical.shape}")
    st.write(f"Final array shape: {input_final_array.shape}")
    st.write(f"Prediction: {prediction}")

    st.success(f"Predicted Obesity Level: {predicted_class}")

