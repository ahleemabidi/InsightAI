import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from termcolor import colored

# Vérifiez que le message de commit est passé en argument
if len(sys.argv) != 2:
    print("Usage: python predict_commit.py '<commit_message>'")
    sys.exit(1)

commit_message = sys.argv[1]

# Chemin vers le fichier CSV
file_path = './.github/workflows/DATA_Finale.csv'

# Charger le fichier CSV
data = pd.read_csv(file_path)

# Charger les modèles et objets nécessaires
rf_model = joblib.load('./rf_model.pkl')
scaler = joblib.load('./scaler.pkl')
label_encoders = joblib.load('./label_encoders.pkl')
le_class = joblib.load('./label_encoder_class.pkl')

# Charger le TF-IDF Vectorizer utilisé pour l'entraînement
vectorizer = TfidfVectorizer(max_features=100)
vectorizer.fit(data['message'])  # Assurez-vous que le vectorizer est formé correctement

# Prétraiter un nouveau commit pour la prédiction
def preprocess_new_commit(commit_text):
    tfidf_vector = vectorizer.transform([commit_text]).toarray()[0]
    
    new_commit = {
        'Date': pd.Timestamp.now(),
        'Duration': 0,
        'User': 'UNKNOWN',
        'Author': 'UNKNOWN',
        'State': 'UNKNOWN',
        'Labels': 'UNKNOWN',
        'Year': pd.Timestamp.now().year,
        'Month': pd.Timestamp.now().month,
        'Day': pd.Timestamp.now().day
    }
    
    for col in label_encoders.keys():
        if new_commit[col] not in label_encoders[col].classes_:
            new_commit[col] = label_encoders[col].classes_[0]
        new_commit[col] = label_encoders[col].transform([new_commit[col]])[0]
    
    for i, value in enumerate(tfidf_vector):
        new_commit[f'tfidf_feature_{i}'] = value
    
    column_names = list(label_encoders.keys()) + [f'tfidf_feature_{i}' for i in range(len(tfidf_vector))]
    missing_cols = set(column_names) - set(new_commit.keys())
    for col in missing_cols:
        new_commit[col] = 0

    new_commit_df = pd.DataFrame([new_commit], columns=column_names)
    new_commit_scaled = scaler.transform(new_commit_df)

    print(f"Processed New Commit Data: {new_commit_df}")
    return new_commit_scaled

# Fonction pour prédire un nouveau commit
def predict_new_commit(commit_text, model_type='rf'):
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        new_commit_prediction_proba = rf_model.predict_proba(new_commit_preprocessed)
    elif model_type == 'nn':
        nn_model = load_model('./nn_model.keras')
        new_commit_prediction_proba = nn_model.predict(new_commit_preprocessed)
    else:
        raise ValueError("Model type not supported")
    
    decoded_classes = le_class.classes_
    prediction_proba = {decoded_classes[i]: new_commit_prediction_proba[0][i] * 100 for i in range(len(decoded_classes))}
    
    formatted_result = "\n".join([
        colored(f"{prob:.2f}% de probabilité que le commit soit classé comme {cls}", 'green')
        for cls, prob in prediction_proba.items()
    ])
    
    return formatted_result

# Afficher les résultats de la prédiction
print(f"Commit Message: {commit_message}")
print(predict_new_commit(commit_message, model_type='rf'))
print(predict_new_commit(commit_message, model_type='nn'))
