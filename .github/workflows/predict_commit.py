
import pandas as pd

# Ajouter la fonction preprocess_new_commit ici
def preprocess_new_commit(commit_text):
    # Convertir le texte de commit en DataFrame (exemple simplifié)
    new_commit_data = {
        'commit_message': [commit_text]
    }
    new_commit_df = pd.DataFrame(new_commit_data)
    
    # Appliquer les mêmes transformations que sur les données d'entraînement
    # Supposons que les mêmes étapes de prétraitement sont nécessaires

    # Encoder les colonnes catégorielles
    for col in categorical_columns:
        if col in new_commit_df:
            le = label_encoders[col]
            new_commit_df[col] = le.transform(new_commit_df[col])
    
    # Normaliser les données
    new_commit_preprocessed = scaler.transform(new_commit_df)
    
    return new_commit_preprocessed

# Le reste du script predict_commit.py
# Ajoutez le reste du script ici à partir de la ligne suivante
# (je suppose qu'il commence par des importations et la définition de la fonction predict_new_commit)

def predict_new_commit(commit_text, model_type='rf'):
    # Appliquez la fonction preprocess_new_commit pour préparer le texte du commit
    new_commit_preprocessed = preprocess_new_commit(commit_text)

    # Prédiction avec le modèle choisi
    if model_type == 'rf':
        model = rf_model
    elif model_type == 'nn':
        model = nn_model
    else:
        raise ValueError(f"Modèle non reconnu : {model_type}")

    prediction = model.predict(new_commit_preprocessed)
    return prediction

# Ajoutez le reste des codes et des fonctions du script original ici
