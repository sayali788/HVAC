import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time

# Streamlit UI
st.title("HVAC Temperature Predictor")
st.sidebar.header("Settings")

# Select Prediction Mode (AI or User Preference)
mode = st.sidebar.radio("Choose Prediction Mode", ("AI", "User Preference"))

# Default value for user preference temperature
user_preference_temp = None

if mode == "User Preference":
    # User Preference Input
    user_preference_temp = st.sidebar.number_input(
        "Enter your preferred desired temperature (°C)", min_value=16.0, max_value=21.0, value=20.0, step=0.5
    )

# File uploader
uploaded_file = st.sidebar.file_uploader("C:/Users/hp/OneDrive/Desktop/Ac_optimization/temperature_data.csv", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Check if required columns are present
        required_columns = ['Indoor Temperature (°C)', 'Outdoor Temperature (°C)']
        if not all(col in data.columns for col in required_columns):
            st.error(f"The uploaded file must contain the following columns: {required_columns}")
        else:
            st.write("### Uploaded Data Preview")
            st.write(data.head())

            # Prepare the data
            X = data[['Indoor Temperature (°C)', 'Outdoor Temperature (°C)']].values

            if mode == "AI" or 'Desired Temperature (°C)' in data.columns:
                # Check if target column exists in AI mode
                y = data['Desired Temperature (°C)'].values if 'Desired Temperature (°C)' in data.columns else None

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

                # Feature scaling
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train the Random Forest model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)

                # Make predictions
                y_pred = rf_model.predict(X_test)

                # Evaluate the model in AI mode
                if mode == "AI":
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"### Model Evaluation")
                    st.write(f"Mean Squared Error: {mse:.2f}")
                    st.write(f"R² Score: {r2:.2f}")

                    # Visualization
                    st.write("### Prediction vs Actual (Indoor Temperature)")
                    plt.figure(figsize=(10, 6))
                    plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
                    plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted')
                    plt.xlabel('Indoor Temperature (Scaled)')
                    plt.ylabel('Desired Temperature')
                    plt.legend()
                    st.pyplot(plt)

            # User Preference Mode
            if mode == "User Preference":
                predicted_temps = []  # List to store predicted temperatures
                adjusted_temps = []  # List for gradual adjustments

                # Simulate predictions for uploaded data
                for i, row in data.iterrows():
                    indoor_temp = row["Indoor Temperature (°C)"]
                    outdoor_temp = row["Outdoor Temperature (°C)"]

                    # Predict temperature using Random Forest
                    predicted_temp = rf_model.predict(scaler.transform([[indoor_temp, outdoor_temp]]))[0]
                    predicted_temps.append(predicted_temp)

                    # Gradual adjustment towards user preference
                    current_temp = predicted_temp
                    while abs(current_temp - user_preference_temp) > 0.1:
                        adjustment = 0.2 if current_temp < user_preference_temp else -0.2
                        current_temp += adjustment
                        adjusted_temps.append(current_temp)
                        time.sleep(0.05)  # Simulate gradual change

                    adjusted_temps.append(current_temp)

                # Add adjusted temperatures to the DataFrame
                data["Predicted Temp"] = predicted_temps
                data["Adjusted Temp"] = adjusted_temps[: len(data)]

                st.write("### Final Adjusted Predictions")
                st.write(data)

                # Allow the user to download the adjusted data
                st.sidebar.download_button(
                    label="Download Adjusted Predictions as CSV",
                    data=data.to_csv(index=False),
                    file_name="adjusted_predictions.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please choose a mode and upload a CSV file to begin.")

