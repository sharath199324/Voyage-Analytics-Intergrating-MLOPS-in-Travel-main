
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import streamlit as st

# Load the trained models and encoders
model_name = joblib.load('model_name.joblib')
model_place = joblib.load('model_place.joblib')
model_price = joblib.load('model_price.joblib')
label_encoder_name = joblib.load('label_encoder_name.joblib')
label_encoder_place = joblib.load('label_encoder_place.joblib')

# Define the prediction function
def predict_hotel(travelCode, userCode, days, price, total):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'travelCode': [travelCode],
        'userCode': [userCode],
        'days': [days],
        'price': [price],
        'total': [total]
    })

    # Make predictions
    predicted_name = model_name.predict(input_data)
    predicted_place = model_place.predict(input_data)
    predicted_price = model_price.predict(input_data)

    # Decode the predictions
    name = label_encoder_name.inverse_transform(predicted_name)[0]
    place = label_encoder_place.inverse_transform(predicted_place)[0]

    return {
        'name': name,
        'place': place,
        'price': predicted_price[0]
    }


# --- Travel App Themed Streamlit UI ---
st.set_page_config(page_title="Hotel Price Predictor", page_icon="🧳", layout="centered")

# Travel-themed header
st.markdown("""
    <div style='text-align:center;'>
        <h1 style='color:#1E90FF;font-family:sans-serif;'>🌴 Hotel Price Predictor 🏨</h1>
        <h4 style='color:#555;'>Your smart travel companion for hotel recommendations</h4>
        <p style='font-size:20px;color:#888;'>Enter your travel details and let us predict your hotel experience!</p>
    </div>
    <hr style='border:1px solid #1E90FF;'>
""", unsafe_allow_html=True)

# Input fields in columns for better layout
col1, col2 = st.columns(2)
with col1:
    travelCode = st.number_input('🛫 Travel Code', min_value=0, step=1)
    days = st.number_input('🗓️ Number of Days', min_value=0, step=1)
    price = st.number_input('💰 Price', min_value=0.0, step=0.01)
with col2:
    userCode = st.number_input('👤 User Code', min_value=0, step=1)
    total = st.number_input('🧾 Total', min_value=0.0, step=0.01)

# Make prediction
if st.button('🚀 Predict My Stay!'):
    prediction = predict_hotel(travelCode, userCode, days, price, total)
    st.markdown("""
        <div style='background:#f0f8ff;border-radius:12px;padding:24px 16px;margin-top:24px;box-shadow:0 2px 8px rgba(30,144,255,0.15);'>
            <h3 style='color:#1E90FF;'>🏨 Hotel Recommendation</h3>
            <ul style='font-size:20px;'>
                <li><b>Hotel Name:</b> <span style='color:#0074D9;'>{}</span></li>
                <li><b>Place:</b> <span style='color:#0074D9;'>{}</span></li>
                <li><b>Predicted Price:</b> <span style='color:#0074D9;'>₹{:.2f}</span></li>
            </ul>
        </div>
    """.format(prediction['name'], prediction['place'], prediction['price']), unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align:center;color:#888;font-size:16px;'>
        ✈️ Happy Travels! | Powered by ML & Streamlit
    </div>
""", unsafe_allow_html=True)

