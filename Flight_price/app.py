import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

scaler_path = os.getenv('SCALER_PATH')
rf_path = os.getenv('RF_PATH')



# Load models once at startup
scaler_model = pickle.load(open(scaler_path, 'rb'))
rf_model = pickle.load(open(rf_path, 'rb'))

def predict_price(input_data, model, scaler):
    df_input2 = pd.DataFrame([input_data])
    X = df_input2
    X = scaler.transform(X)
    y_prediction = model.predict(X)
    return y_prediction[0]


# --- Custom CSS for Tourism Theme ---
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1500&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main {
        background: rgba(255,255,255,0.85);
        border-radius: 16px;
        padding: 2rem 2rem 1rem 2rem;
        margin-top: 2rem;
        box-shadow: 0 6px 32px 0 rgba(0,0,0,0.15);
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px 32px;
        margin-top: 12px;
    }
    .stButton>button:hover {
        background-color: #0074D9;
        color: #fff;
    }
    .result-box {
        background: #f0f8ff;
        border-radius: 12px;
        padding: 30px 18px;
        margin-top: 28px;
        box-shadow: 0 2px 8px rgba(30,144,255,0.15);
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 16px;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Custom Header ---
st.markdown(
    """
    <div style='text-align:center;'>
        <h1 style='color:#1E90FF;font-family:sans-serif;'>Flight Price Predictor</h1>
        <h4 style='color:#555;'>Your smart travel companion for the best flight deals</h4>
        <p style='font-size:20px;color:#888;'>Enter your travel details and let us predict your journey cost!</p>
    </div>
    <hr style='border:1px solid #1E90FF;'>
    """, unsafe_allow_html=True
)


# --- Input fields with travel icons and tooltips ---
col1, col2 = st.columns(2)
with col1:
    from_ = st.selectbox('From: Boarding City', [
        'Aracaju', 'Brasilia', 'Campo_Grande', 'Florianopolis', 'Natal', 'Recife', 'Rio_de_Janeiro', 'Salvador', 'Sao_Paulo'
    ], help='Select your departure city')
    flightType = st.selectbox('Class: Flight Class', ['economic', 'firstClass', 'premium'], help='Choose your travel class')
    week_no = st.number_input('Week Number', min_value=1, max_value=53, value=7, help='Week number of travel (1-53)')
with col2:
    Destination = st.selectbox('To: Destination City', [
        'Aracaju', 'Brasilia', 'Campo_Grande', 'Florianopolis', 'Natal', 'Recife', 'Rio_de_Janeiro', 'Salvador', 'Sao_Paulo'
    ], help='Select your arrival city')
    agency = st.selectbox('Agency', ['Rainbow', 'CloudFy', 'FlyingDrops'], help='Select your booking agency')
    week_day = st.number_input('Week Day', min_value=1, max_value=7, value=5, help='Day of week (1=Mon, 7=Sun)')
day = st.number_input('Day of Month', min_value=1, max_value=31, value=5, help='Day of the month you plan to travel')

# --- Predict Button ---
predict_btn = st.button('Predict My Flight Price!')

if predict_btn:
    # Prepare one-hot encoded input as before
    boarding = 'from_' + from_
    boarding_city_list = [
        'from_Florianopolis (SC)',
        'from_Sao_Paulo (SP)',
        'from_Salvador (BH)',
        'from_Brasilia (DF)',
        'from_Rio_de_Janeiro (RJ)',
        'from_Campo_Grande (MS)',
        'from_Aracaju (SE)',
        'from_Natal (RN)',
        'from_Recife (PE)'
    ]
    destination = 'destination_' + Destination
    destination_city_list = [
        'destination_Florianopolis (SC)',
        'destination_Sao_Paulo (SP)',
        'destination_Salvador (BH)',
        'destination_Brasilia (DF)',
        'destination_Rio_de_Janeiro (RJ)',
        'destination_Campo_Grande (MS)',
        'destination_Aracaju (SE)',
        'destination_Natal (RN)',
        'destination_Recife (PE)'
    ]
    selected_flight_class = 'flightType_' + flightType
    class_list = ['flightType_economic', 'flightType_firstClass', 'flightType_premium']
    selected_agency = 'agency_' + agency
    agency_list = ['agency_Rainbow', 'agency_CloudFy', 'agency_FlyingDrops']
    travel_dict = dict()
    for city in boarding_city_list:
        travel_dict[city] = 1 if city[:-5] == boarding else 0
    for city in destination_city_list:
        travel_dict[city] = 1 if city[:-5] == destination else 0
    for flight_class in class_list:
        travel_dict[flight_class] = 1 if flight_class == selected_flight_class else 0
    for ag in agency_list:
        travel_dict[ag] = 1 if ag == selected_agency else 0
    travel_dict['week_no'] = week_no
    travel_dict['week_day'] = week_day
    travel_dict['day'] = day
    try:
        predicted_price = round(predict_price(travel_dict, rf_model, scaler_model), 2)
        st.markdown(f"""
            <div class='result-box'>
                <h3 style='color:#1E90FF;'>Estimated Flight Price</h3>
                <p style='font-size:28px;color:#0074D9;'><b>Rs. {predicted_price}</b></p>
                <p style='color:#555;'>Bon voyage! Plan your trip with confidence.</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Footer ---
st.markdown(
    """
    <hr>
    <div class='footer'>
        "Travel is the only thing you buy that makes you richer." &nbsp;|&nbsp; Powered by ML & Streamlit
    </div>
    """, unsafe_allow_html=True
)
