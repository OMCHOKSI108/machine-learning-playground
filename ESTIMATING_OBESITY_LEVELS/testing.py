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

# Test with sample inputs
def test_prediction(gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans):
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

    # Scale numerical
    scaled_numerical = scaler.transform(input_data[numerical_cols])

    # Encode categorical
    encoded_categorical = encoder.transform(input_data[categorical_cols])

    # Combine
    input_final_array = np.hstack([scaled_numerical, encoded_categorical])

    # Predict
    prediction = model.predict(input_final_array)
    predicted_class = prediction[0]

    print(f"Input: Gender={gender}, Age={age}, Height={height}, Weight={weight}, ...")
    print(f"Predicted Obesity Level: {predicted_class}")
    print(f"Final array shape: {input_final_array.shape}")
    print(f"Prediction: {prediction}")
    return predicted_class

# Run tests with different inputs
print("Testing Model Predictions:")
print("=" * 50)

# Test 1: Default values
test_prediction('Male', 25, 170, 70, 'no', 'no', 2, 3, 'Sometimes', 'no', 2, 'no', 1, 1, 'Sometimes', 'Public_Transportation')

print("\n" + "=" * 50)

# Test 2: Different values
test_prediction('Female', 50, 160, 80, 'yes', 'yes', 3, 4, 'Always', 'yes', 3, 'yes', 0, 0, 'Frequently', 'Automobile')

print("\n" + "=" * 50)

# Test 3: Another set
test_prediction('Male', 30, 180, 90, 'yes', 'yes', 1, 2, 'no', 'no', 1, 'no', 2, 2, 'no', 'Walking')

print("\nTesting complete.")