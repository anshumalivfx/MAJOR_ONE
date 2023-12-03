from flask import Flask, request, jsonify
import torch
import numpy as np
import gensim
from absa import model, preprocessing, infer_processing, best_aspects
import numpy as np
import pandas as pd
import sqlite3

app = Flask(__name__)

# Load the LSTM model
lstm_model = torch.load("My-Model.pt", map_location=torch.device('cpu'))

# Load the FastText model
fasttext_model = gensim.models.FastText.load("FastText-Model-For-ABSA.bin")
# Define aspects
c


@app.route("/average_rating", methods=['GET'])
def average_rating():
    con = sqlite3.connect("flipkart_products.db")
    items = pd.read_sql_query("SELECT * from items", con)
    con.close()
    con = sqlite3.connect("flipkart_products.db")
    df = pd.read_sql_query("SELECT * from ECMB000001", con)
    for i in range(2, len(items) + 1):

        df_temp = pd.read_sql_query("SELECT * from ECMB{:06d}".format(i), con)
        df = pd.concat([df, df_temp])
    con.close()
    return jsonify({'average_rating': df['ratings'].mean()})


@app.route("/number_of_rows", methods=['GET'])
def number_of_rows():
    con = sqlite3.connect("flipkart_products.db")
    items = pd.read_sql_query("SELECT * from items", con)
    con.close()
    con = sqlite3.connect("flipkart_products.db")
    df = pd.read_sql_query("SELECT * from ECMB000001", con)
    for i in range(2, len(items) + 1):

        df_temp = pd.read_sql_query("SELECT * from ECMB{:06d}".format(i), con)
        df = pd.concat([df, df_temp])
    con.close()
    
    return jsonify({'number_of_rows': len(df)})
    

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        # Get the input text from the request
        data = request.get_json()
        text = data['text']

        # Preprocess the text and get the best aspect
        processed_text = preprocessing(text)
        best_aspect = best_aspects(processed_text, aspects)

        # Infer processing and get the sentiment
        processed_input = infer_processing(text).to(torch.device('cpu'))
        model.eval()
        sentiment = model(processed_input)
        sentiment = sentiment.cpu().detach().numpy()[0]

        # Convert sentiment to Positive or Negative
        if sentiment > 0.5:
            sentiment_label = 'Positively'
        else :
            sentiment_label = 'Negatively'

        # Prepare the response
        response = {
            'aspect': best_aspect,
            'sentiment': sentiment_label
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_value', methods=['GET'])
def negative_and_positive():
    con = sqlite3.connect("flipkart_products.db")
    items = pd.read_sql_query("SELECT * from items", con)
    con.close()
    con = sqlite3.connect("flipkart_products.db")
    df = pd.read_sql_query("SELECT * from ECMB000001", con)
    for i in range(2, len(items) + 1):

        df_temp = pd.read_sql_query("SELECT * from ECMB{:06d}".format(i), con)
        df = pd.concat([df, df_temp])
    con.close()
    df.dropna(inplace=True, axis=0)
    # negative or positive based on ratings
    df['ratings'] = df['ratings'].astype(int)
    df['sentiment'] = np.where(df['ratings'] > 3, 'Positive', 'Negative')
    response_data = {
    "name": "Predicted Positive and Negative Reviews",
    "data": [
        int(df[df["sentiment"] == "Positive"]["sentiment"].count()),
        int(df[df["sentiment"] == "Negative"]["sentiment"].count())
    ]
    }


# Rest of your code...

    return jsonify(response_data)

@app.route("/dashboardTableData", methods=["GET"])
def dashboardTableData():
    conn = sqlite3.connect("flipkart_products.db")
    df = pd.read_sql_query("SELECT * from items", conn)
    conn.close()
    df.dropna(inplace=True, axis=0)



@app.route("/login")
def login():
    return "haha"

if __name__ == '__main__':
    app.run(debug=True)
