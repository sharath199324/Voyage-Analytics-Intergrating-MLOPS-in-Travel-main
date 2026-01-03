# This code is part of a Flask application
from flask import Flask, request, jsonify
import pickle  # For model serialization
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

# Initialize the sentence transformer model
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

scaler_path = os.getenv('SCALER_PATH')
pca_path = os.getenv('PCA_PATH')
logistic_path = os.getenv('LOGISTIC_PATH')
#company_encoder_path = os.getenv('company_encoder_path')

scaler_model = pickle.load(open(scaler_path, 'rb'))
pca_model = pickle.load(open(pca_path, 'rb'))
logistic_model = pickle.load(open(logistic_path, 'rb'))
#label_encoder = pickle.load(open(company_encoder_path, 'rb'))

def predict_price(input_data, lr_model, pca, scaler):
    # Prepare the input data
    text_columns = ['name']

    # Initialize an empty DataFrame
    df = pd.DataFrame([input_data])
    
    #filtering records based on relavent categories in the target variable
    #df=df[(df['gender']=='male') | (df['gender']=='female') ]
    
    
    # Encode userCode and company to numeric values
    label_encoder = LabelEncoder()

    df['company_encoded'] = label_encoder.fit_transform(df['company'])
    #df['gender_encoded'] = label_encoder.fit_transform(df['gender'])
    
    # Encode text-based columns and create embeddings
    for column in text_columns:
        df[column + '_embedding'] = df[column].apply(lambda text: model.encode(text))

    # Apply PCA separately to each text embedding column
    n_components = 23  # Adjust the number of components as needed
    text_embeddings_pca = np.empty((len(df), n_components * len(text_columns)))

    for i, column in enumerate(text_columns):
        embeddings = df[column + '_embedding'].values.tolist()
        embeddings_pca = pca.transform(embeddings)
        text_embeddings_pca[:, i * n_components:(i + 1) * n_components] = embeddings_pca

    # Combine text embeddings with other numerical features if available
    numerical_features = ['code','company_encoded','age']
    

    X_numerical = df[numerical_features].values

    # Combine PCA-transformed text embeddings and numerical features
    X = np.hstack((text_embeddings_pca, X_numerical))

    # Scale the data using the same scaler used during training
    X = scaler.transform(X)

    # Make predictions using the trained Linear Regression model
    y_pred = lr_model.predict(X)

    return y_pred[0]



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Gender Classification Model</title>
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f9f9f9;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        h1 {
            color: #007BFF;
            font-size: 36px;
            margin-bottom: 20px;
        }

        form {
            text-align: left;
        }

        input[type="text"],
        input[type="number"],
        select {
            width: 100%;
            padding: 15px;
            margin: 15px 0;
            border: none;
            border-bottom: 2px solid #007BFF;
            font-size: 18px;
            background-color: transparent;
            color: #333;
            transition: border-bottom 0.3s ease;
        }

        input[type="text"]:focus,
        input[type="number"]:focus,
        select:focus {
            border-bottom: 2px solid #0056b3;
            outline: none;
        }

        input[type="checkbox"],
        input[type="radio"] {
            margin-right: 10px;
        }

        input[type="submit"] {
            background-color: #007BFF;
            color: #fff;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 20px;
            transition: background-color 0.3s ease;
            margin-top: 20px;
        }

        input[type="submit"]:hover {
            background-color: #0056b3;
        }

        p#prediction {
            margin-top: 20px;
            font-size: 24px;
            color: #007BFF;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gender Classification Model</h1>
        <form action="/predict" method="POST">
            <label for="Username">Username:</label>
            <input type="text" name="Username" placeholder="Enter name of traveller" value="Charlotte Johnson">
            
            <label for="Usercode">Usercode:</label>
            <input type="number" name="Usercode" min="0.00" max="1339.00" placeholder="Enter the user id of traveller">

            <label for="Traveller_Age">Traveller_Age:</label>
            <input type="number" name="Traveller_Age" min="21" max="65" placeholder="Enter the age of traveller">

            <label for="company_name">Company name:</label>
            <select name="company_name">
                <option value="Acme Factory">Acme Factory</option>
                <option value="Wonka Company">Wonka Company</option>
                <option value="Monsters CYA">Monsters CYA</option>
                <option value="Umbrella LTDA">Umbrella LTDA</option>
                <option value="4You">4You</option>
            </select>

            <input type="submit" value="Predict">
        </form>
        <p id="prediction"></p>
    </div>
</body>
</html>"""


    


@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        usercode = request.form.get('Usercode')
        company = request.form.get('company_name')
        name = request.form.get('Username')
        age = request.form.get('Traveller_Age')


        # Create a dictionary to store the input data
        data = {
            'code': usercode,
            'company': company,
            'name': name,
            'age': age,
           
        }

        # Perform prediction using the custom_input dictionary
        prediction = predict_price(data, logistic_model, pca_model, scaler_model)
        
        if prediction ==0:
            gender='female'
        else:
            gender='male'
        
        prediction = str(gender)
       

        return jsonify({'prediction':  prediction})


if __name__ == "__main__":
    import subprocess
    import requests
    import time

    # Get ngrok auth token and port from .env
    # ngrok_auth_token = os.getenv('NGROK_AUTH_TOKEN')
    flask_port = int(os.getenv('FLASK_PORT', 1111))
    # ngrok_file_path = os.getenv('NGROK_FILE_PATH')

    # Add ngrok auth token (idempotent)
    # subprocess.run([ngrok_file_path, "config", "add-authtoken", ngrok_auth_token], check=True)

    # Start ngrok tunnel
    # ngrok_proc = subprocess.Popen([ngrok_file_path, "http", str(flask_port)])
    # time.sleep(3)  # Wait for ngrok to initialize
    # try:
    #     tunnel_url = requests.get("http://localhost:4040/api/tunnels").json()["tunnels"][0]["public_url"]
    #     print(f" * ngrok tunnel running at: {tunnel_url}")
    # except Exception as e:
    #     print(" * ngrok started, but could not fetch public URL. Is ngrok running?")

    try:
        app.run(host="0.0.0.0", port=flask_port, debug=True)
    finally:
        # if 'ngrok_proc' in locals() and ngrok_proc:
        #     ngrok_proc.terminate()
        #     print("ngrok tunnel closed.")
        pass