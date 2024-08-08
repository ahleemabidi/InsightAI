import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import sys

# Catégorie personnalisée pour remplacer la catégorie 0
CUSTOM_CATEGORY = "Catégorie Personnalisée"

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

# Hyperparameter tuning pour Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
rf_grid.fit(X_train_sm, y_train_sm)

# Meilleurs paramètres et score
print("Best Parameters for Random Forest: ", rf_grid.best_params_)
print("Best Score for Random Forest: ", rf_grid.best_score_)

# Créer et entraîner le modèle Random Forest avec les meilleurs paramètres
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train_sm, y_train_sm)

# Faire des prédictions et évaluer le modèle Random Forest
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr')

print("Confusion Matrix (Random Forest):")
print(conf_matrix_rf)

print("\nClassification Report (Random Forest):")
print(class_report_rf)

print("\nAccuracy Score (Random Forest):")
print(accuracy_rf)

print("\nROC AUC Score (Random Forest):")
print(roc_auc_rf)

# Créer et entraîner le modèle de réseau de neurones
y_train_sm_one_hot = to_categorical(y_train_sm)
y_test_one_hot = to_categorical(y_test)

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_sm.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(len(le_class.classes_), activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_sm, y_train_sm_one_hot, epochs=50, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Sauvegarder le modèle Keras
nn_model.save('./nn_model.h5')

# Faire des prédictions et évaluer le modèle de réseau de neurones
y_pred_nn = nn_model.predict(X_test)
y_pred_proba_nn = y_pred_nn

y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)

conf_matrix_nn = confusion_matrix(y_test, y_pred_nn_classes)
class_report_nn = classification_report(y_test, y_pred_nn_classes)
accuracy_nn = accuracy_score(y_test, y_pred_nn_classes)
roc_auc_nn = roc_auc_score(y_test, y_pred_proba_nn, multi_class='ovr')

print("Confusion Matrix (Neural Network):")
print(conf_matrix_nn)

print("\nClassification Report (Neural Network):")
print(class_report_nn)

print("\nAccuracy Score (Neural Network):")
print(accuracy_nn)

print("\nROC AUC Score (Neural Network):")
print(roc_auc_nn)

# Sauvegarder les modèles et les objets nécessaires
joblib.dump(rf_model, './rf_model.pkl')
joblib.dump(scaler, './scaler.pkl')
joblib.dump(label_encoders, './label_encoders.pkl')

# Prétraiter une nouvelle commit pour la prédiction
def preprocess_new_commit(commit_text):
    # Créer un dictionnaire simulé pour la nouvelle commit en utilisant des valeurs connues
    new_commit = {
        'Date': pd.Timestamp.now(),
        'Duration': 0,  # Exemple: durée en secondes
        'User': data['User'].mode()[0],  # Utiliser la valeur la plus fréquente comme exemple
        'Author': data['Author'].mode()[0],  # Utiliser la valeur la plus fréquente comme exemple
        'State': data['State'].mode()[0],  # Utiliser la valeur la plus fréquente comme exemple
        'Labels': data['Labels'].mode()[0],  # Utiliser la valeur la plus fréquente comme exemple
        'Year': pd.Timestamp.now().year,
        'Month': pd.Timestamp.now().month,
        'Day': pd.Timestamp.now().day
    }

    # Encoder les colonnes catégorielles avec une gestion des labels inconnus
    for col in categorical_columns:
        if new_commit[col] not in label_encoders[col].classes_:
            new_commit[col] = label_encoders[col].transform([new_commit[col]])[0]
        else:
            new_commit[col] = label_encoders[col].transform([new_commit[col]])[0]

    # Ajouter toutes les colonnes manquantes avec des valeurs par défaut
    missing_cols = set(column_names) - set(new_commit.keys())
    for col in missing_cols:
        new_commit[col] = 0

    # Réordonner les colonnes selon l'ordre attendu
    new_commit_df = pd.DataFrame([new_commit], columns=column_names)

    # Normaliser les données
    new_commit_scaled = scaler.transform(new_commit_df)

    print(f"Processed New Commit Data: {new_commit_df}")
    return new_commit_scaled

# Fonction pour prédire une nouvelle commit
def predict_new_commit(commit_text, model_type='rf'):
    # Prétraiter la nouvelle commit
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        # Charger le modèle Random Forest
        model = joblib.load('./rf_model.pkl')
        # Faire la prédiction avec le modèle Random Forest
        new_commit_prediction_proba = model.predict_proba(new_commit_preprocessed)
    elif model_type == 'nn':
        # Charger le modèle de réseau de neurones
        model = load_model('./nn_model.h5')
        # Faire la prédiction avec le modèle de réseau de neurones
        new_commit_prediction_proba = model.predict(new_commit_preprocessed)
    else:
        raise ValueError("Model type not supported")
    
    # Décoder les classes
    decoded_classes = le_class.inverse_transform(np.arange(len(le_class.classes_)))
    
    # Mapping pour les noms des classes avec la catégorie personnalisée
    class_mapping = {i: (CUSTOM_CATEGORY if i == 0 else decoded_classes[i]) for i in range(len(decoded_classes))}
    
    # Afficher les résultats
    result_strings = []
    for i, proba in enumerate(new_commit_prediction_proba[0]):
        class_name = class_mapping.get(i, "Unknown")
        result_strings.append(f"{proba*100:.2f}% de probabilité que le commit soit classé comme {class_name}.")
    
    print("Prediction Probabilities: ", new_commit_prediction_proba)
    return "\n".join(result_strings)

# Lire le message du commit depuis les arguments de la ligne de commande
if len(sys.argv) > 1:
    commit_message = sys.argv[1]
    model_type = 'rf'  # Par défaut, utiliser Random Forest
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    # Effectuer la prédiction avec le modèle spécifié
    result = predict_new_commit(commit_message, model_type=model_type)
    # Afficher le résultat avec un formatage clair
    print("\n====================\n")
    print(f"**Résultat de la Prédiction:** {result}")
    print("\n====================\n")
else:
    print("Veuillez fournir un message de commit en argument.")
