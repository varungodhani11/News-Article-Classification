# News Article Classification (Fake vs Real)

A machine-learning project that classifies news article text as **Fake** or **Real** using NLP and a Streamlit web interface.

> **Important:** This is a text-classification demo, not a fact-checking service. Predictions and confidence scores reflect patterns learned from the dataset and do not prove whether an article is true.

## Project Overview

The project uses labeled fake and real news articles to train and evaluate text-classification models. Article text is cleaned, transformed with TF-IDF, and passed to machine-learning classifiers.

## Tools and Technologies

- Python
- Pandas
- NLTK
- scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Joblib
- Jupyter Notebook

## Workflow

1. Load `Fake.csv` and `True.csv`.
2. Assign `Fake` and `Real` labels and combine the datasets.
3. Inspect data, remove duplicate rows, and handle empty text.
4. Clean and preprocess article text using regular expressions and NLTK.
5. Split data into training and test sets with stratification.
6. Convert text into TF-IDF features.
7. Train Logistic Regression and Multinomial Naive Bayes.
8. Evaluate using accuracy, precision, recall, F1-score, and confusion matrices.
9. Compare model metrics.
10. Save the Logistic Regression model and TF-IDF vectorizer with Joblib.
11. Run a Streamlit app for interactive predictions.

## Results

Evaluation on the held-out test set (8,936 samples):

| Model | Accuracy | Fake F1-score | Real F1-score |
|---|---:|---:|---:|
| Logistic Regression | 98.76% | 0.9881 | 0.9870 |
| Multinomial Naive Bayes | 93.91% | 0.9420 | 0.9359 |

Logistic Regression scored higher on this test split. Results may differ on other data, and dataset-specific patterns can affect performance.

## Project Structure

```text
News Article Classification/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── notebook/
│   └── News_Article_Classification.ipynb
├── models/
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

The `screenshots/` folder is intentionally omitted because it was removed while empty. You can add it later if you decide to include app screenshots.

## Setup and Run

Run these commands from the project root directory.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Start the Streamlit app

```bash
cd app
streamlit run app.py
```

## Dataset

The project uses the Kaggle Fake and Real News Dataset:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `Fake.csv` and `True.csv` inside the `data/` folder for the documented project structure. The notebook and app paths should match your local folder layout.

## Limitations

- Dataset labels are not independent verification of every article.
- High test accuracy does not guarantee performance on new sources, topics, or time periods.
- The model learns text patterns; its prediction or confidence should not be treated as proof of factual accuracy.
