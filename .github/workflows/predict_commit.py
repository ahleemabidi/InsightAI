import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from tensorflow.keras.models import load_model
import os

# Charger le fichier CSV
file_path = './DATA_Finale.csv'
data = pd.read_csv(file_path)

# Convertir les colonnes de date/heure en datetime
data['Date'] = pd.to_datetime(data['Date'])
data['Created At'] = pd.to_datetime(data['Created At'])
data['Updated At'] = pd.to_datetime(data['Updated At'])

# Extraire les fonctionnalités pertinentes des colonnes de date/heure
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Convertir la durée en secondes
data['Duration'] = pd.to_timedelta(data['Duration']).dt.total_seconds()

# Fusionner les classes rares
data['Classification'] = data['Classification'].replace({
    'DOCUMENTATION (A NE PAS PRENDRE EN COMPTE)': 'OTHER',
    'BUG (PRIORITAIRE sur les autres LABELS)': 'BUG',
    'ÉVOLUTION (PRIORITAIRE sur les autres LABELS)': 'ÉVOLUTION'
})

# Encoder les colonnes catégorielles
label_encoders = {}
categorical_columns = ['User', 'Author', 'State', 'Labels']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encoder la colonne Classification
le_class = LabelEncoder()
data['Classification'] = le_class.fit_transform(data['Classification'])

# Prétraiter les données
X = data.drop(['Classification', 'Date', 'commit', 'message', 'functions', 'Created At', 'Updated At'], axis=1, errors='ignore')
y = data['Classification']

# Sauvegarder les noms de colonnes
column_names = X.columns

# Normaliser les données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ajuster dynamiquement k_neighbors
def get_k_neighbors(y_train):
    min_class_samples = min(y_train.value_counts())
    return max(1, min_class_samples - 1)

k_neighbors = get_k_neighbors(y_train)

# Appliquer SMOTE pour équilibrer les classes sur l'ensemble d'entraînement
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Créer et entraîner le modèle Random Forest avec les meilleurs paramètres
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, n_jobs=-1)
rf_grid.fit(X_train_sm, y_train_sm)

rf_model = rf_grid.best_estimator_
rf_model.fit(X_train_sm, y_train_sm)

# Sauvegarder le modèle et les encodeurs
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(le_class, 'label_encoder_class.pkl')

# Fonction pour prédire une nouvelle commit
def predict_new_commit(commit_text, model_type='rf'):
    # Prétraiter la nouvelle commit
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        # Faire la prédiction avec le modèle Random Forest
        new_commit_prediction_proba = rf_model.predict_proba(new_commit_preprocessed)

    # Décoder les classes
    decoded_classes = le_class.inverse_transform(np.arange(len(le_class.classes_)))
    
    # Mapping pour les noms des classes
    class_mapping = {i: decoded_classes[i] for i in range(len(decoded_classes))}
    
    # Afficher les résultats
    result_strings = []
    for i, proba in enumerate(new_commit_prediction_proba[0]):
        class_name = class_mapping.get(i, "Unknown")
        result_strings.append(f"{proba*100:.2f}% de probabilité que le commit soit classé comme {class_name}.")
    
    print("Prediction Probabilities: ", new_commit_prediction_proba)
    return "\n".join(result_strings)

# Fonction pour prétraiter une nouvelle commit
def preprocess_new_commit(commit_text):
    # Créer un DataFrame pour la nouvelle commit
    new_commit_data = {'commit_message': [commit_text]}
    new_commit_df = pd.DataFrame(new_commit_data)
    
    # Encoder les colonnes catégorielles
    for col in categorical_columns:
        if col in new_commit_df:
            le = label_encoders[col]
            new_commit_df[col] = le.transform(new_commit_df[col])
    
    # Normaliser les données
    new_commit_preprocessed = scaler.transform(new_commit_df)
    
    return new_commit_preprocessed

# Chemin complet pour le répertoire courant du script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Utiliser la dernière commit pour la prédiction
with open(os.path.join(script_dir, '.git/COMMIT_EDITMSG'), 'r') as file:
    commit_message = file.read().strip()

# Effectuer la prédiction avec le modèle RF par défaut
result = predict_new_commit(commit_message, model_type='rf')

# Afficher le résultat
print(result)
