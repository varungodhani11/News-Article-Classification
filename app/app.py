
import streamlit as st
import joblib
import re
import string
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure the Streamlit page
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered"
)

# Define project paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"

# Load the saved model and TF-IDF vectorizer
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

# Load English stopwords
@st.cache_resource
def load_stop_words():
    words = set(stopwords.words("english"))
    return words - {"no", "not", "nor"}

# Clean news text using the training preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    stop_words = load_stop_words()

    cleaned_tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(cleaned_tokens)

# Main application interface
st.title("📰 News Article Classifier")
st.write(
    "Enter a news article below to get a Fake or Real prediction."
)

news_input = st.text_area(
    "Paste your news article:",
    height=250,
    placeholder="Enter the news title and article text here..."
)

if st.button("Classify News", type="primary"):
    if not news_input.strip():
        st.warning("Please enter a news article first.")
    else:
        try:
            model, vectorizer = load_model()

            cleaned_news = clean_text(news_input)

            if not cleaned_news:
                st.warning(
                    "No usable text found. Please enter a longer article."
                )
            else:
                news_tfidf = vectorizer.transform([cleaned_news])

                prediction = model.predict(news_tfidf)[0]
                probabilities = model.predict_proba(news_tfidf)[0]

                class_names = list(model.classes_)
                predicted_index = class_names.index(prediction)
                confidence = probabilities[predicted_index] * 100

                st.subheader("Prediction")

                if prediction == "Fake":
                    st.error("Prediction: Fake")
                else:
                    st.success("Prediction: Real")

                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.2f}%"
                )

                st.progress(float(confidence / 100))

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not proof that the article is true or false."
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")