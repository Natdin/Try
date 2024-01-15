# your code goes here
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
import streamlit as st

# Load the data
df = pd.read_csv('/kaggle/input/tour-pack/tour_packages.csv')

# Extract the features and target
X = df[['Place', 'Month', 'Traveler', 'Agency', 'PackageType']]
y = df['Price']

# One-hot encode the categorical features
ohe = OneHotEncoder(handle_unknown='ignore')
X_ohe = ohe.fit_transform(X)

# Train the model
lr = LinearRegression()
lr.fit(X_ohe, y)

# Save the model
joblib.dump(lr, 'tour_price_prediction_model.joblib')
joblib.dump(ohe, 'tour_price_prediction_encoder.joblib')

# Define the web app
def predict_price():
    # Load the saved model and encoder
    lr = joblib.load('tour_price_prediction_model.joblib')
    ohe = joblib.load('tour_price_prediction_encoder.joblib')
    
    # Define the dropdown menus for input values
    place_dropdown = st.selectbox('Place:', ['Select'] + list(sorted(df['Place'].unique())))
    month_dropdown = st.selectbox('Month:', ['Select'] + list(sorted(df['Month'].unique())))
    traveler_dropdown = st.selectbox('Traveler:', ['Select'] + [1, 2, 3, 4, 5])
    agency_dropdown = st.selectbox('Agency:', ['Select'] + list(sorted(df['Agency'].unique())))
    package_type_dropdown = st.selectbox('Package Type:', ['Select'] + list(sorted(df['PackageType'].unique())))

    # Make prediction when Predict button is clicked
    if st.button('Predict'):
        # Check if all dropdown menus have been selected
        if (place_dropdown == 'Select') or (month_dropdown == 'Select') or \
           (traveler_dropdown == 'Select') or (agency_dropdown == 'Select') or (package_type_dropdown == 'Select'):
            st.write('Please select all dropdown menus.')
        else:
            # Encode the input using the same encoder used for training
            X_input = pd.DataFrame([[place_dropdown, month_dropdown, traveler_dropdown, agency_dropdown, package_type_dropdown]], columns=X.columns)
            X_input_ohe = ohe.transform(X_input)
            # Make prediction
            prediction = lr.predict(X_input_ohe)[0]
            # Print the predicted price
            st.write("Predicted price: Rs.", int(prediction))

# Run the web app
if __name__ == '__main__':
    predict_price()
