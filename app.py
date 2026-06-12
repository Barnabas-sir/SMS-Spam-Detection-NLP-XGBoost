import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# NLP setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# App title
st.title("📩 SMS Spam Detection System")

st.markdown(
    """
    Enter a message below and the trained XGBoost model
    will classify it as Spam or Ham.
    """
)

message = st.text_area(
    "Enter Message"
)

if st.button("Predict"):

    if len(message.strip()) == 0:
        st.warning("Please enter a message.")

    else:

        processed = preprocess_text(message)

        text_features = tfidf.transform([processed])

        engineered_features = np.array([[
            len(message),
            len(message.split()),
            sum(c.isdigit() for c in message),
            sum(c.isupper() for c in message),
            message.count("!")
        ]])

        X = hstack([
            text_features,
            engineered_features
        ])

        prediction = model.predict(X)[0]

        probability = model.predict_proba(X)[0][1]

        if prediction == 1:
            st.error("🚨 Spam Detected")
        else:
            st.success("✅ Legitimate Message")

        st.metric(
            "Spam Probability",
            f"{probability:.2%}"
        )