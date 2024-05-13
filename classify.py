import argparse
import sqlite3
import pandas as pd
import tensorflow as tf
import joblib
import os
import json
import re
import chardet
import numpy as np
from lime.lime_text import LimeTextExplainer

# Define paths
model_path = "tv_genre_model_multi.h5"
vectorizer_path = 'vectorizer_multi.pkl'

def load_genre_data():
    connection = sqlite3.connect("tvmaze.sqlite")
    genre_df = pd.read_sql_query("SELECT * FROM tvmaze_genre", connection)
    connection.close()
    return genre_df['genre'].unique()

genre_columns = load_genre_data()

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

class TVGenreClassifier:
    def __init__(self, model_path, vectorizer_path, genre_columns,threshold=0.5):
        self.model = tf.keras.models.load_model(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.genre_columns = genre_columns
        self.threshold = threshold
        
    def preprocess(self, descriptions):
        cleaned_descriptions = [clean_text(desc) for desc in descriptions]
        return self.vectorizer.transform(cleaned_descriptions).toarray()
    
    def predict(self, descriptions):
        processed_data = self.preprocess(descriptions)
        predictions = self.model.predict(processed_data)
        return predictions
    
    def get_genres(self, predictions):
        predicted_genres = []
        for pred in predictions:
            genres = [self.genre_columns[i] for i, value in enumerate(pred) if value >= self.threshold]
            predicted_genres.append(genres)
        return predicted_genres
    
    def predict_proba(self, texts):
        
        processed_texts = self.preprocess([clean_text(text) for text in texts])
        predictions = self.model.predict(processed_texts)
        return predictions  # Ensure this is a 2D array

# Read the input text file with the detected encoding
def read_input_text(input_file_path):
    encoding = detect_file_encoding(input_file_path)
    with open(input_file_path, 'r', encoding=encoding) as file:
        input_text = file.read()
    return input_text

# Explain the prediction with LIME
def explain_with_lime(input_text, classifier, num_features=5, output_html='lime_explanation.html'):
    explainer = LimeTextExplainer(class_names=classifier.genre_columns)
    exp = explainer.explain_instance(input_text, classifier.predict_proba, num_features=num_features)
    exp.save_to_file(output_html)

def main(input_file_path, output_json_file, encoding, explanation_output_dir):
    # Set default encoding if not provided
    encoding = encoding or detect_file_encoding(input_file_path)

    # Read the input text
    with open(input_file_path, 'r', encoding=encoding) as file:
        input_text = file.read()

    # Create an instance of the TVGenreClassifier with a threshold
    classifier = TVGenreClassifier(model_path, vectorizer_path, genre_columns, threshold=0.5)

    # Get predictions and save to JSON
    predictions = classifier.predict([input_text])
    genres = classifier.get_genres(predictions)[0]
    with open(output_json_file, 'w', encoding='UTF-8') as json_file:
        json.dump(genres, json_file, ensure_ascii=False)

    # Explain the prediction with LIME
    if explanation_output_dir:
        # Create the directory if it does not exist
        if not os.path.exists(explanation_output_dir):
            os.makedirs(explanation_output_dir)
        
        # Define the output path for the HTML explanation
        explanation_html = os.path.join(explanation_output_dir, 'lime_explanation.html')
    else:
        # If no directory is provided, use the current working directory
        explanation_html = 'lime_explanation.html'
    
    # Generate the explanation with LIME
    explain_with_lime(input_text, classifier, output_html=explanation_html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify TV show genres based on description")
    parser.add_argument("--input-file", required=True, help="Path to the input file with TV show description")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for genres")
    parser.add_argument("--encoding", help="Input file encoding")
    parser.add_argument("--explanation-output-dir", help="Directory for explanation output")

    args = parser.parse_args()
    main(args.input_file, args.output_json_file, args.encoding, args.explanation_output_dir)