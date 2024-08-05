import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model

# Simulated example variables, these should be properly defined elsewhere in your code
categorical_columns = ['commit_message']
label_encoders = {col: LabelEncoder() for col in categorical_columns}
scaler = StandardScaler()
rf_model = RandomForestClassifier()
nn_model = load_model('path_to_your_neural_network_model')

def preprocess_new_commit(commit_text):
    new_commit_data = {'commit_message': [commit_text]}
    new_commit_df = pd.DataFrame(new_commit_data)
    
    for col in categorical_columns:
        if col in new_commit_df:
            le = label_encoders[col]
            new_commit_df[col] = le.transform(new_commit_df[col])
    
    new_commit_preprocessed = scaler.transform(new_commit_df)
    
    return new_commit_preprocessed

def predict_new_commit(commit_text, model_type='rf'):
    new_commit_preprocessed = preprocess_new_commit(commit_text)

    if model_type == 'rf':
        model = rf_model
    elif model_type == 'nn':
        model = nn_model
    else:
        raise ValueError(f"Modèle non reconnu : {model_type}")

    prediction = model.predict(new_commit_preprocessed)
    return prediction

if __name__ == "__main__":
    commit_message = "Example commit message"  # Remplacez par le message de commit réel
    model_type = 'rf'  # Ou 'nn' selon le modèle que vous voulez utiliser
    result = predict_new_commit(commit_message, model_type=model_type)

    print(f"Prediction results: {result}")
