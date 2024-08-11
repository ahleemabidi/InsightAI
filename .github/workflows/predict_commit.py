import sys
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from termcolor import colored

# Load saved models and preprocessors
rf_model = joblib.load('./rf_model.pkl')
scaler = joblib.load('./scaler.pkl')
label_encoders = joblib.load('./label_encoders.pkl')
le_class = joblib.load('./label_encoder_class.pkl')
vectorizer = joblib.load('./vectorizer.pkl')

# Preprocess a new commit for prediction
def preprocess_new_commit(commit_text):
    # Transform the commit message with TF-IDF
    tfidf_vector = vectorizer.transform([commit_text]).toarray()[0]

    # Create a default dictionary with placeholder values for all features
    new_commit = {
        'Date': pd.Timestamp.now(),
        'Duration': 0,  # Example: duration in seconds
        'User': 'UNKNOWN',
        'Author': 'UNKNOWN',
        'State': 'UNKNOWN',
        'Labels': 'UNKNOWN',
        'Year': pd.Timestamp.now().year,
        'Month': pd.Timestamp.now().month,
        'Day': pd.Timestamp.now().day
    }

    # Encode categorical columns with handling for unknown labels
    for col in ['User', 'Author', 'State', 'Labels']:
        if new_commit[col] not in label_encoders[col].classes_:
            new_commit[col] = label_encoders[col].classes_[0]
        new_commit[col] = label_encoders[col].transform([new_commit[col]])[0]

    # Add TF-IDF features to the dictionary
    for i, value in enumerate(tfidf_vector):
        new_commit[f'tfidf_feature_{i}'] = value

    # Create a DataFrame for the new commit
    new_commit_df = pd.DataFrame([new_commit])

    # Normalize the data
    new_commit_scaled = scaler.transform(new_commit_df)

    print(f"Processed New Commit Data: {new_commit_df}")
    return new_commit_scaled

# Function to predict a new commit
def predict_new_commit(commit_text, model_type='rf'):
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        model = rf_model
        new_commit_prediction_proba = model.predict_proba(new_commit_preprocessed)
    elif model_type == 'nn':
        model = load_model('./nn_model.keras')
        new_commit_prediction_proba = model.predict(new_commit_preprocessed)
    else:
        raise ValueError("Model type not supported")
    
    print("Raw Prediction Probabilities: ", new_commit_prediction_proba)
    
    decoded_classes = le_class.classes_
    prediction_proba = {decoded_classes[i]: new_commit_prediction_proba[0][i] * 100 for i in range(len(decoded_classes))}
    
    # Create formatted and colorful output
    formatted_result = "\n".join([
        colored(f"{prob:.2f}% de probabilité que le commit soit classé comme {cls}", 'green')
        for cls, prob in prediction_proba.items()
    ])
    
    return formatted_result

# Get commit message from command line argument
if len(sys.argv) != 2:
    print("Usage: python predict_commit.py 'commit message'")
    sys.exit(1)

commit_message = sys.argv[1]
result = predict_new_commit(commit_message)
print(result)
