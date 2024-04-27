import streamlit as st
import joblib
import numpy as np

# Load the Model that we will be using
xgb_model = joblib.load('xgb_model.pkl')

gender_dict = {'Male': [0,0], 'Female': [1,0]}


def predict(input_data):
    features = np.array([input_data]).reshape(1,-1)
    prediction = xgb_model.predict(features)
    return prediction[0]


# Web app
st.title('Calories Burned Prediction')

Gender = st.sidebar.selectbox('Gender', options=['Male', 'Female'])
Age = st.number_input('Age', min_value=10, max_value=100, step=1, value=30)
Height = st.number_input('Height in cm', min_value=100, max_value=25000, step=1, value=170)
Weight = st.number_input('Weight in kg', min_value=30, max_value=200, step=1, value=70)
Duration = st.number_input('Duration of exercise in minutes', min_value=1, max_value=900, step=1, value=30)
Heart_Rate = st.number_input('Heart Rate', min_value=30, max_value=220, step=1, value=80)
Body_temperature = st.number_input('Body temperature in celsius', min_value=35, max_value=50, step=1, value=37)

encoded_gender = gender_dict[Gender]

input_data = encoded_gender + [Age, Height, Weight, Duration, Heart_Rate, Body_temperature]

if st.button('Predict'):
    result = predict(input_data)
    st.write(f"Your calories burned prediction is {result:.2f} calories")
