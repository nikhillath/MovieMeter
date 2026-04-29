import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

from tensorflow.keras.models import load_model

NUM_WORDS = 10000
INDEX_FROM = 3
START_CHAR = 1
OOV_CHAR = 2

word_index = imdb.get_word_index()
model = load_model('my_model.h5')
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
max_review_length = 200

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - INDEX_FROM, '?') for i in encoded_review])

def preprocess_text(text):
    tokens = text_to_word_sequence(text)
    encoded_review = []
    for word in tokens:
        word_id = word_index.get(word)
        if word_id is None:
            encoded_review.append(OOV_CHAR)
            continue

        word_id += INDEX_FROM
        if word_id >= NUM_WORDS:
            encoded_review.append(OOV_CHAR)
        else:
            encoded_review.append(word_id)

    encoded_review = [START_CHAR] + encoded_review
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_review_length)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='positive' if prediction[0][0] >= 0.5 else 'negative'
    return sentiment, prediction[0][0]

import streamlit as st 
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

user_input = st.text_area("Movie Review", height=200)
if st.button("Predict Sentiment"):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] >= 0.5 else 'negative'
    st.write(f"Predicted Sentiment: {sentiment} (Confidence: {prediction[0][0]:.2f})")

else:
    st.write("Please enter a movie review and click the button to predict its sentiment.")
