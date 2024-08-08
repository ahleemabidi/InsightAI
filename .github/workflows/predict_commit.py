import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys

# Charger le fichier CSV
file_path = './.github/workflows/DATA_Finale.csv'
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

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Distribution des classes avant et après SMOTE
print("Distribution des classes avant SMOTE:")
print(y_train.value_counts())

# Application de SMOTE pour traiter le déséquilibre des classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Distribution des classes après SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Entraîner un modèle de régression (Random Forest dans ce cas)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Sauvegarder le modèle et les encoders
model_path = './rf_model.pkl'
joblib.dump(rf_model, model_path)
joblib.dump(scaler, './scaler.pkl')
joblib.dump(label_encoders, './label_encoders.pkl')

# Évaluer les performances du modèle sur l'ensemble de test
y_pred = rf_model.predict(X_test)
print("Rapport de classification sur l'ensemble de test:")
print(classification_report(y_test, y_pred))
print("Matrice de confusion:")
print(confusion_matrix(y_test, y_pred))

# Fonction pour prédire un nouveau commit
def predict_new_commit(commit_message, model_type='rf'):
    if model_type == 'rf':
        model = joblib.load(model_path)
    else:
        raise ValueError("Model type not supported")
    
    # Prétraiter la nouvelle commit
    new_commit_preprocessed = preprocess_new_commit(commit_message)
    
    # Faire la prédiction
    prediction = model.predict(new_commit_preprocessed)
    
    # Décoder la prédiction
    result = le_class.inverse_transform(prediction)
    
    return result[0]

# Fonction pour prétraiter une nouvelle commit
def preprocess_new_commit(commit_text):
    # Créer un DataFrame pour la nouvelle commit
    new_commit_data = {'commit_message': [commit_text]}
    new_commit_df = pd.DataFrame(new_commit_data)
    
    # Ajouter les colonnes manquantes avec des valeurs par défaut
    for col in column_names:
        if col not in new_commit_df.columns:
            new_commit_df[col] = 0
    
    # Réorganiser les colonnes pour correspondre à l'ordre des colonnes du modèle
    new_commit_df = new_commit_df[column_names]
    
    # Normaliser les données
    new_commit_preprocessed = scaler.transform(new_commit_df)
    
    return new_commit_preprocessed

# Lire le message du commit depuis les arguments de la ligne de commande
if len(sys.argv) > 1:
    commit_message = sys.argv[1]
    # Effectuer la prédiction avec le modèle RF par défaut
    result = predict_new_commit(commit_message, model_type='rf')
    # Afficher le résultat avec un formatage clair
    print("\n====================\n")
    print(f"**Résultat de la Prédiction:** {result}")
    print("\n====================\n")
else:
    print("Veuillez fournir un message de commit en argument.")
