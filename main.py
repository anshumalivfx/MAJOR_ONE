import streamlit as st
import torch
import pandas as pd
import numpy as np
import re
import demoji
import unicodedata as uni
from langdetect import detect
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pad_sequence
from absa import model, preprocessing, infer_processing, best_aspects, preprocess_text, tokenizer, maxlen, pad_sequences
import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import networkx as nx
import pydot
from plotly import graph_objects as go



names = ["Lakshita Gupta", "Bhavey Mittal", "Anshumali Karna"]
usernames = ["lakshita", "bhavey", "anshumali"]

file_path = Path(__file__).parent / "hashed_passwords.pkl"



with open(file_path, "rb") as f:
    hashed_passwords = pickle.load(f)


authenticator = stauth.Authenticate(names=names, usernames=usernames, passwords=hashed_passwords,
                                    cookie_name="analytics_dashboard",
                                    key="abcd",
                                    cookie_expiry_days=1
                                    )

# authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "app_home", "auth")

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Incorrect username or password. Please try again.")

if authentication_status == None:
    st.warning("Please log in to continue.")
    
if authentication_status == True:

    # ... (The rest of the imports remain the same)
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
    # Set up Streamlit


# Logout 

    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    # Load the pre-trained model and vocabulary
    st.title("Flipkart Reviews Analysis")
    # model_path = "My-ModelABSA.pt"
    # state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # # Check for and handle unexpected keys
    # unexpected_keys = [k for k in state_dict.keys() if k not in model.state_dict()]
    # missing_keys = [k for k in model.state_dict() if k not in state_dict]

    # # Handle unexpected keys (remove them or handle them appropriately)
    # for key in unexpected_keys:
    #     state_dict.pop(key)

    # # Handle missing keys (initialize them or handle them appropriately)
    # for key in missing_keys:
    #     state_dict[key] = torch.randn_like(model.state_dict()[key])
    
    # model.load_state_dict(state_dict)
    # unexpected_keys = [k for k in state_dict.keys() if k not in model.state_dict()]
    # missing_keys = [k for k in model.state_dict() if k not in state_dict]
    # model.eval()
    aspects = ["phone", "camera", "battery", "neutral", "processor", "build quality", "software", "display", "design", "storage"]
    # Function to preprocess and get sentiment
    @st.cache_data
    def get_sentiment(text):
        # Preprocess the text and get the best aspect
        processed_text = preprocessing(text)
        best_aspect = best_aspects(processed_text, aspects)

        # Infer processing and get the sentiment
        test_data = [text]
        test_sequences = tokenizer.texts_to_sequences(test_data)
        test_padded = pad_sequences(test_sequences, padding='post', maxlen=maxlen)
        pred = model.predict(test_padded)
        # Convert sentiment to Positive or Negative
        if pred[0] > 0.5:
            sentiment_label = 'Positively'
        else:
            sentiment_label = 'Negatively'

        return sentiment_label, best_aspect

    # Function to apply sentiment analysis to the dataset
    # perform sentiment analysis on the dataset randomly selected 100 reviews from the dataset
    def analyze_dataset(df):
        # Select 100 random reviews
        df = df.sample(100)

        # Apply sentiment analysis
        results = []
        for index, row in df.iterrows():
            text = row['review']
            sentiment, aspect = get_sentiment(text)
            results.append([text, sentiment, aspect])

        return results
    
    def create_aspect_graph(results):
        G = nx.Graph()

        # Add nodes and edges based on aspects in each review
        for _, sentiment, aspect in results:
            aspects_list = aspect.split(', ')
            for i in range(len(aspects_list)):
                for j in range(i + 1, len(aspects_list)):
                    # Add edges between aspects that co-occur in a review
                    if G.has_edge(aspects_list[i], aspects_list[j]):
                        G[aspects_list[i]][aspects_list[j]]['weight'] += 1
                    else:
                        G.add_edge(aspects_list[i], aspects_list[j], weight=1)

        return G

    # ... (The code to read the dataset remains the same)

    # Display the analyzed dataset
    st.title("Sentiment Analysis on Dataset")
    st.write("Original Dataset:")
    st.write(df)

    # Analyze the dataset
    results = analyze_dataset(df)

    # Display the results
    st.write("Sentiment Analysis Results:")
    st.table(pd.DataFrame(results, columns=['Review', 'Sentiment', 'Aspect']))

    sentiment_df = pd.DataFrame(results, columns=['Review', 'Sentiment', 'Aspect'])
    # st.table(sentiment_df[""])


    st.title("Sentiment Distribution")
    sentiment_counts = sentiment_df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    positive_text = " ".join(sentiment_df[sentiment_df['Sentiment'] == 'Positively']['Aspect'])
    negative_text = " ".join(sentiment_df[sentiment_df['Sentiment'] == 'Negatively']['Aspect'])

    review_lengths = df['review'].apply(lambda x: len(x.split()))
    st.title("Review Length Distribution")
    
    fig, ax = plt.subplots()
    ax.hist(review_lengths, bins=20)
    st.pyplot(fig)
    # Pie Chart for Sentiment Distribution
    st.title("Sentiment Distribution")
    sentiment_counts = sentiment_df["Sentiment"].value_counts()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig_pie)

    st.title("Word Cloud of Reviews")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(sentiment_df['Review']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
    
    # Bar Chart for Aspect Distribution
    st.title("Aspect Distribution")
    aspect_counts = sentiment_df["Aspect"].value_counts()
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    aspect_counts.plot(kind='bar', ax=ax_bar)
    ax_bar.set_xlabel("Aspect")
    ax_bar.set_ylabel("Count")
    st.pyplot(fig_bar)

    st.title("Machine Learning Model Information")

    # Example: Display accuracy and other relevant metrics
    model_accuracy = 0.93  # Replace with your model's accuracy
    st.write(f"Model Accuracy: {model_accuracy:.2%}")

    # Example: Display information about the model architecture
    model_architecture = "LSTM"  # Replace with your model's architecture
    st.write(f"Model Architecture: {model_architecture}")

    # Example: Display information about the training dataset
    training_dataset_size = 5000  # Replace with the actual size of your training dataset
    st.write(f"Training Dataset Size: {training_dataset_size} reviews")
    
    
