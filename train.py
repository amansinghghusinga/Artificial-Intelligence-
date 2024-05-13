import sqlite3
import pandas as pd
import tensorflow as tf
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Set seed for reproducibility
import numpy as np
np.random.seed(42)
tf.random.set_seed(42)

def load_data(training_data):
    connection = sqlite3.connect(training_data)
    shows_df = pd.read_sql_query("SELECT * FROM tvmaze", connection)
    genre_df = pd.read_sql_query("SELECT * FROM tvmaze_genre", connection)
    connection.close()
    return genre_df, shows_df

def preprocess_data(genre_df, shows_df):
    merged_df = pd.merge(shows_df, genre_df, on='tvmaze_id', how='left')
    merged_df.dropna(subset=['genre'], inplace=True)
    merged_df['description'].fillna("", inplace=True)
    genre_columns = genre_df['genre'].unique()
    genre_indicators_df = pd.crosstab(index=merged_df['tvmaze_id'], columns=merged_df['genre']).reset_index()
    final_df = pd.merge(shows_df, genre_indicators_df, on='tvmaze_id', how='inner')
    final_df = final_df.dropna(subset=['description'])
    final_df['description'] = final_df['description'].apply(clean_text)
    return final_df, genre_columns

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def vectorize_and_split_data(final_df, genre_columns):
    max_tokens = 10000
    output_sequence_length = 120
    vectorizer = TfidfVectorizer(max_features=max_tokens)
    X = vectorizer.fit_transform(final_df['description']).toarray()
    if X.shape[1] < output_sequence_length:
        zeros = np.zeros((X.shape[0], output_sequence_length - X.shape[1]))
        X = np.hstack([X, zeros])
    X_temp, X_test, y_temp, y_test = train_test_split(X, final_df[genre_columns].values)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

def train_and_save_model(X_train, X_val, y_train, y_val, genre_columns, vectorizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(genre_columns), activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32, 
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)])
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    model.save("tv_genre_model_multi.h5")
    joblib.dump(vectorizer, 'vectorizer_multi.pkl')
    return train_accuracy, val_accuracy

if __name__ == "__main__":
    training_data = "tvmaze.sqlite"
    genre_df, shows_df = load_data(training_data)
    final_df, genre_columns = preprocess_data(genre_df, shows_df)
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = vectorize_and_split_data(final_df, genre_columns)
    train_accuracy, val_accuracy = train_and_save_model(X_train, X_val, y_train, y_val, genre_columns, vectorizer)
    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
