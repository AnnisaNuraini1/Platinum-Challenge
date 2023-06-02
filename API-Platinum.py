import csv
import json
import logging
import os
# menyimpan model
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
import re
import sqlite3
from logging.handlers import RotatingFileHandler

import nltk
# lib standart
import numpy as np
import pandas as pd
import requests
# import library untuk model neural network
import tensorflow as tf
from flasgger import LazyJSONEncoder, LazyString, Swagger, swag_from
from flask import (Flask, jsonify, render_template, render_template_string,
                   request)
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
# bagian dari pipe line untuk handling kolom
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# menghitung nilai f1
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
# untuk tuning hyperparameter
# Untuk split data
from sklearn.model_selection import GridSearchCV, train_test_split
# Untuk Algoritma ML yang akan di pakai
from sklearn.neural_network import MLPClassifier
# untuk membangun pipeline ML
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessingtextpackage.normalisasitext import text_preprocessing

# # untuk impute nilai kosong
# from sklearn.impute import SimpleImputer

# # untuk scale data
# from sklearn.preprocessing import StandardScaler

# # untuk one hot data kategori
# from sklearn.preprocessing import OneHotEncoder

# # untuk pca
# from sklearn.decomposition import PCA


# Download the Indonesian stopwords if not already downloaded
nltk.download("stopwords")

# Define the maximum sequence length
MAX_SEQUENCE_LENGTH = 100

app = Flask(__name__)
# Konfigurasi logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.DEBUG)
app.logger.addHandler(handler)


###############################################################################################################
app.json_encoder = LazyJSONEncoder

swagger_template = dict(
    info={
        "title": LazyString(
            lambda: "API Documentation for Data Processing and Modeling"
        ),
        "version": LazyString(lambda: "1.0.0"),
        "description": LazyString(
            lambda: "Dokumentasi API untuk Data Processing dan Modeling"
        ),
    },
    host=LazyString(lambda: request.host),
)

swagger_config = {
    "headers": [],
    "specs": [{"endpoint": "docs", "route": "/docs.json"}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)
###############################################################################################################
# GET


@app.route("/")
def home():
    return render_template("1_index.html")


@swag_from("docs/get_tweet.yml", methods=["GET"])
@app.route("/upload", methods=["GET"])
def get_api_upload():
    try:
        with sqlite3.connect("platinum1.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM upload_analisis_sentimen")
            rows = cursor.fetchall()

        data_list = []
        for row in rows:
            data_dict = {
                "id": row[0],
                "flag": row[1],
                "sentimen": row[2],
                "tweet_baru": row[3],
                "tweet": row[4],
            }
            data_list.append(data_dict)

        json_response = {
            "status_code": 200,
            "description": "Mendapatkan Data dari Database",
            "data": data_list,
        }

        return jsonify(json_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

####################################################################################################################################################################################
# INPUT NN


@swag_from("docs/post_nn.yml", methods=["POST"])
@app.route("/input_NN", methods=["POST"])
def nn_text():
    input_json = request.get_json(force=True)
    text = input_json["Tweet"]
    Hasil = text_preprocessing(text)  # memproses keseluruhan teks

    count_vect = pickle.load(open("feature_New5.sav", "rb"))
    model_NN = pickle.load(open("model_NN5.sav", "rb"))

    text_transformed = count_vect.transform([Hasil])
    sentimen = model_NN.predict(text_transformed)[0]

    id = None

    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        query_text = "INSERT INTO analisis_sentimen (id, tweet, tweet_baru, sentimen, flag) VALUES (?, ?, ?, ?, ?)"
        val = (id, text, Hasil, sentimen, "Neural Network")
        cursor.execute(query_text, val)
        conn.commit()

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis by using Neural Network",
        "data": {
            "id": id,
            "flag": "Neural Network",
            "sentiment": sentimen,
            "tweet_baru": Hasil,
            "Tweet": text,
        },
    }

    response_data = jsonify(json_response)
    return response_data


# UPLOAD NN

@swag_from("docs/upload_nn.yml", methods=["POST"])
@app.route("/upload", methods=["POST"])
def uploadDoc():
    # Mengunggah file CSV
    file = request.files["file"]

    try:
        data = pd.read_csv(file, encoding="iso-8859-1", on_bad_lines="skip")
    except:
        data = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")

    count_vect = pickle.load(open("feature_New5.sav", "rb"))
    model_NN = pickle.load(open("model_NN5.sav", "rb"))

    # Menyimpan data ke dalam database SQLite
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        for index, row in data.iterrows():
            text = row["Tweet"]
            Hasil = text_preprocessing(text)
            text_transformed = count_vect.transform([Hasil])
            sentimen = model_NN.predict(text_transformed)[0]
            id = None

            # Menyimpan data ke dalam database SQLite
            df = pd.DataFrame({"id": [None], "tweet": [text]})
            df.to_sql("Neural_Network", conn, if_exists="append", index=False)
            cursor.execute(
                "INSERT INTO upload_analisis_sentimen (id, flag, sentimen, tweet_baru, tweet) VALUES (?, ?, ?, ?, ?)",
                (None, "Neural Network", sentimen, Hasil, text,),
            )

        # Mengambil data dari database
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM upload_analisis_sentimen")
        rows = cursor.fetchall()

    # Format data sesuai kebutuhan
    data_list = []
    for row in rows:
        data_dict = {
            "id": row[0],
            "flag": row[1],
            "sentimen": row[2],
            "tweet_baru": row[3],
            "tweet": row[4],
        }
        data_list.append(data_dict)

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis by using Neural Network",
        "data": data_list,
    }

    response_data = jsonify(json_response)
    return response_data


#################################################################################################################################################################################################################
# input lstm


@swag_from("docs/lstm_text_input.yml", methods=["POST"])
@app.route("/LSTM_text_input", methods=["POST"])
def lstm_text_input():
    data = request.get_json()
    text = data['Tweet']
    hasil = text_preprocessing(text)  # Preprocess the input text

    # Load the pre-trained model
    loaded_model_lstm = load_model(r'lstm4.h5')

    # Initialize tokenizer
    tokenizer = Tokenizer()

    # Fit tokenizer
    # Pass the preprocessed text to initialize the tokenizer
    tokenizer.fit_on_texts([hasil])

    # Transform the preprocessed text
    text_sequence = tokenizer.texts_to_sequences([hasil])
    text_transformed = pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Predict sentiment
    Sentiment = loaded_model_lstm.predict(text_transformed)[0]
    # Mengubah nilai sentimen menjadi keterangan negatif, positif, atau netral
    max_index = np.argmax(Sentiment)
    if max_index == 0:
        sentiment_label = "Negative"
    elif max_index == 1:
        sentiment_label = "Neutral"
    else:
        sentiment_label = "Positive"
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        query_text = "INSERT INTO analisis_sentimen (id, tweet, tweet_baru, sentimen, flag) VALUES (?, ?, ?, ?, ?)"
        id = None
        val = (id, text, hasil, sentiment_label, "LSTM")
        cursor.execute(query_text, val)
        conn.commit()

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis by using LSTM",
        "data": {
            "id": id,  # Replace id with the actual ID value
            "flag": "LSTM",
            "sentiment": sentiment_label,
            "tweet_baru": hasil,
            "tweet": text,
        },
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/lstm_text_upload.yml", methods=["POST"])
@app.route("/LSTM_text_Upload", methods=["POST"])
def uploadDoc_lstm():
    # Mengunggah file CSV
    file = request.files["file"]

    try:
        data = pd.read_csv(file, encoding="iso-8859-1", on_bad_lines="skip")
    except:
        data = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")

    # Load the pre-trained model
    loaded_model_lstm = load_model(r'lstm4.h5')

    # Initialize tokenizer
    tokenizer = TfidfVectorizer()

    # Menyimpan data ke dalam database SQLite
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        for index, row in data.iterrows():
            text = row['Tweet']
            hasil = text_preprocessing(text)  # Preprocess the input text

            # Fit tokenizer
            # Pass the preprocessed text to initialize the tokenizer
            tokenizer.fit([hasil])

            # Transform the preprocessed text
            text_sequence = tokenizer.transform([hasil])
            text_transformed = text_sequence.toarray()

            # Predict sentiment
            Sentiment = loaded_model_lstm.predict(text_transformed)[0]

            # Mengubah nilai sentimen menjadi keterangan negatif, positif, atau netral
            max_index = np.argmax(Sentiment)
            if max_index == 0:
                sentiment_label = "Negative"
            elif max_index == 1:
                sentiment_label = "Neutral"
            else:
                sentiment_label = "Positive"

            # Insert data into database
            df = pd.DataFrame({"id": [None], "tweet": [text]})
            df.to_sql("LSTM", conn, if_exists="append", index=False)
            cursor.execute("INSERT INTO upload_analisis_sentimen (id, flag, sentimen, tweet_baru, tweet) VALUES (?, ?, ?, ?, ?)",
                           (None, "LSTM", sentiment_label, hasil, text))

    # Mengambil data dari database
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM upload_analisis_sentimen")
        rows = cursor.fetchall()

    # Format data sesuai kebutuhan
    data_list = []
    for row in rows:
        data_dict = {
            "id": row[0],
            "flag": row[1],
            "sentimen": row[2],
            "tweet_baru": row[3],
            "tweet": row[4],
        }
        data_list.append(data_dict)

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis by using LSTM",
        "data": data_list,
    }

    response_data = jsonify(json_response)
    return response_data


#####################################################################################################################################################

# UPdate dan Delete NN


@swag_from("docs/put_nn.yml", methods=["PUT"])
@app.route("/PUT/<int:id_input_NN>", methods=["PUT"])
def update_tweet_id_NN(id_input_NN):
    input_json = request.get_json(force=True)
    text = input_json["Tweet"]
    Hasil = text_preprocessing(text)  # memproses keseluruhan teks

    count_vect = pickle.load(open("feature_New5.sav", "rb"))
    model_NN = pickle.load(open("model_NN5.sav", "rb"))

    text_transformed = count_vect.transform([Hasil])
    sentimen = model_NN.predict(text_transformed)[0]

    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE analisis_sentimen SET id =?, flag = ?, sentimen = ?, tweet_baru = ?, tweet =? WHERE id = ?",
            (id_input_NN, "Neural Network", sentimen, Hasil, text, id_input_NN),
        )
        conn.commit()

    response = {"id": id_input_NN,
                "flag": "Neural Network",
                "sentimen": sentimen,
                "tweet_baru": Hasil,
                "tweet": text}
    return json.dumps(response)


@swag_from("docs/delete_nn.yml", methods=["DELETE"])
@app.route("/DELETE/<int:id_input_NN>", methods=["DELETE"])
def delete_tweet_id_NN(id_input_NN):
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM analisis_sentimen WHERE id = ?", (id_input_NN,))
        conn.commit()

    response = {"status": "success delete"}
    return json.dumps(response)


#################################################################################################################################################

# PUT dan DELETE LSTM


@swag_from("docs/delete_lstm.yml", methods=["DELETE"])
@app.route("/DELETE/<int:id_input_LSTM>", methods=["DELETE"])
def delete_tweet_id_LSTM(id_input_LSTM):
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM analisis_sentimen WHERE id = ?", (id_input_LSTM,))
        conn.commit()

    response = {"status": "success delete"}
    return json.dumps(response)


@swag_from("docs/put_lstm.yml", methods=["PUT"])
@app.route("/put_lstm/<id>", methods=["PUT"])
def lstm_PUT(id):
    data = request.get_json()
    text = data['Tweet']
    hasil = text_preprocessing(text)  # Preprocess the input text

    # Load the pre-trained model
    loaded_model_lstmpath = r"D:\A BINAR ACADEMY BOOTCAMP\COURSE\LEVEL PLATINUM\_full_Challenge Platinum_Annisa_\lstm4.h5"
    loaded_model_lstm = load_model(loaded_model_lstmpath)

    # Initialize tokenizer
    tokenizer = TfidfVectorizer()

    # Fit tokenizer
    # Pass the preprocessed text to initialize the tokenizer
    tokenizer.fit([hasil])

    # Transform the preprocessed text
    text_sequence = tokenizer.transform([hasil])
    text_transformed = text_sequence.toarray()

    # Predict sentiment
    Sentiment = loaded_model_lstm.predict(text_transformed)[0]
    # Mengubah nilai sentimen menjadi keterangan negatif, positif, atau netral
    max_index = np.argmax(Sentiment)
    if max_index == 0:
        sentiment_label = "Negative"
    elif max_index == 1:
        sentiment_label = "Neutral"
    else:
        sentiment_label = "Positive"
    with sqlite3.connect("platinum1.db") as conn:
        cursor = conn.cursor()
        query_text = "UPDATE analisis_sentimen SET tweet=?, tweet_baru=?, sentimen=?, flag=? WHERE id=?"
        val = (text, hasil, sentiment_label, "LSTM", id)
        cursor.execute(query_text, val)
        conn.commit()

    json_response = {
        "status_code": 200,
        "description": "Result of Sentiment Analysis by using LSTM",
        "data": {
            "id": id,
            "flag": "LSTM",
            "sentiment": sentiment_label,
            "tweet_baru": hasil,
            "tweet": text,
        },
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == "__main__":
    # app.debug = True
    app.run()


# run flask otomatis debug
# flask --app test_demo_swag --debug run


# import json

# testing api

# data = {"Tweet": "saya benci kamu karena kamu bego dan sangat jelek"}
# json_object = json.dumps(data)
# r = requests.post(url="http://127.0.0.1:5000/gold", data=json_object)
# print(r.text)
