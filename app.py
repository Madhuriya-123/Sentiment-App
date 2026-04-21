import streamlit as st
import pickle

with open('binary_sentiment_model.pkl', 'rb') as f:
    binary_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = binary_model.predict(text_vectorized)
    return 'Positive' if prediction[0] == 1 else 'Negative'

st.title('Sentiment Analysis App')
user_input = st.text_area('Enter your text here:')

if st.button('Predict Sentiment'):
    if user_input:
        result = predict_sentiment(user_input)
        st.write(f'Sentiment: {result}')
    else:
        st.write('Please enter some text to analyze.')