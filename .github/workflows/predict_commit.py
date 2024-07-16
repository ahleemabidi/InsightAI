import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMote
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle

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
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
rf_grid.fit(X_train_sm, y_train_sm)

rf_model = rf_grid.best_estimator_
rf_model.fit(X_train_sm, y_train_sm)

# Créer et entraîner le modèle de réseau de neurones
y_train_sm_one_hot = to_categorical(y_train_sm)
y_test_one_hot = to_categorical(y_test)

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_sm.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(len(le_class.classes_), activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_sm, y_train_sm_one_hot, epochs=50, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Fonction pour prédire une nouvelle commit
def predict_new_commit(commit_text, model_type='rf'):
    # Prétraiter la nouvelle commit
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        # Faire la prédiction avec le modèle Random Forest
        new_commit_prediction_proba = rf_model.predict_proba(new_commit_preprocessed)
    elif model_type == 'nn':
        # Faire la prédiction avec le modèle de réseau de neurones
        new_commit_prediction_proba = nn_model.predict(new_commit_preprocessed)

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

# Utiliser la dernière commit pour la prédiction
with open('./.git/COMMIT_EDITMSG', 'r') as file:
    commit_message = file.read().strip()

# Effectuer la prédiction
result = predict_new_commit(commit_message, model_type='rf')

# Sauvegarder le résultat dans un fichier texte
with open('prediction_results.txt', 'w') as file:
    file.write(result)
