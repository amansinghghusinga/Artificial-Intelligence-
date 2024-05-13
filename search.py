import argparse
import json
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(input_file, output_json_file, encoding='UTF-8'):
    try:
        # Load Data and Vectorizer
        df_tvmaze = pd.read_csv("tvmaze_data.csv")
        with open("tfidf_vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)
        tfidf_matrix = vectorizer.transform(df_tvmaze['text'])
        
        # Read Search Query from File
        with open(input_file, "r", encoding=encoding) as file:
            query = file.read().strip()

        # Search Logic
        query_as_vector = vectorizer.transform([query])
        similarities = cosine_similarity(tfidf_matrix, query_as_vector)
        ranked_results = np.argsort(similarities, axis=0)[::-1]

        top_matches = []
        for result_position in ranked_results[:3]:
            show_number = result_position[0]
            scoring = similarities[show_number]
            if scoring == 0.0:
                break
            show_info = df_tvmaze.iloc[show_number]
            top_matches.append({
                "tvmaze_id": int(show_info['tvmaze_id']),
                "showname": show_info['showname']
            })

        # Save the results to the output JSON file
        with open(output_json_file, 'w', encoding=encoding) as json_file:
            json.dump(top_matches, json_file, ensure_ascii=False)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")

    args = parser.parse_args()

    search(args.input_file, args.output_json_file, args.encoding)
