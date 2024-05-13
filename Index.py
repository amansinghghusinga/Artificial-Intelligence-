import argparse
import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def index_data(sqlite_file):
    try:
        df = fetch_data_from_db(sqlite_file)
        df_tvmaze = preprocess_data(df)
        X, vectorizer = vectorize_text(df_tvmaze)
        save_data(df_tvmaze, vectorizer)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def fetch_data_from_db(sqlite_file):
    # Connect to the SQLite database
    connection = sqlite3.connect(sqlite_file)

    # Read the tvmaze table into a dataframe
    query = """
    SELECT tvmaze_id, showname, description
    FROM tvmaze
    """
    df_tvmaze = pd.read_sql(query, connection)

    return df_tvmaze


def preprocess_data(df_tvmaze):
    # Replace NaN and 'N/A' values in the 'description' column with empty strings
    df_tvmaze['description'].fillna("", inplace=True)
    df_tvmaze['description'].replace('N/A', "", inplace=True)

    # Drop rows where the 'description' column contains empty strings
    df_tvmaze_filtered = df_tvmaze[df_tvmaze['description'] != ""]

    # Remove rows without TV show names
    df_tvmaze = df_tvmaze_filtered[df_tvmaze_filtered['showname'].notna() & (df_tvmaze_filtered['showname'] != '')]

    # Text Cleaning
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = ''.join([char for char in text if char.isalpha() or char.isspace()])
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    df_tvmaze['description'] = df_tvmaze['description'].apply(clean_text)

    # Concatenate showname and description for vectorization
    df_tvmaze['text'] = df_tvmaze['showname'] + " " + df_tvmaze['description']

    return df_tvmaze


def vectorize_text(df_tvmaze):
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_tvmaze['text'])
    return X, vectorizer


def save_data(df_tvmaze, vectorizer):
    # Save the vectorizer
    vectorizer_filename = "tfidf_vectorizer.pkl"
    with open(vectorizer_filename, "wb") as f:
        pickle.dump(vectorizer, f)

    # Save the cleaned DataFrame
    df_tvmaze.to_csv("tvmaze_data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index data from an SQLite database")
    parser.add_argument("--raw-data", required=True, help="Path to the SQLite database file")

    args = parser.parse_args()

    index_data(args.raw_data)
