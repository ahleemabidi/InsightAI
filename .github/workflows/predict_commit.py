import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from termcolor import colored

# Charger les données
file_path = './.github/workflows/DATA_Finale.csv'
data = pd.read_csv(file_path)

# Convertir les colonnes date/heure en datetime
data['Date'] = pd.to_datetime(data['Date'])
data['Created At'] = pd.to_datetime(data['Created At'])
data['Updated At'] = pd.to_datetime(data['Updated At'])

# Extraire des caractéristiques pertinentes des colonnes date/heure
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

# Encoder les colonnes catégoriques
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
column_names = list(X.columns)

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

# Appliquer SMOTE pour équilibrer les classes dans l'ensemble d'entraînement
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Tuning des hyperparamètres pour Random Forest
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

# Afficher les résultats du modèle Random Forest
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

# Sauvegarder les modèles et objets nécessaires
joblib.dump(rf_model, './rf_model.pkl')
joblib.dump(scaler, './scaler.pkl')
joblib.dump(label_encoders, './label_encoders.pkl')
joblib.dump(le_class, './label_encoder_class.pkl')  # Sauvegarder l'encodeur de labels pour classification

# Créer et adapter le vectoriseur TF-IDF sur les messages de commit
vectorizer = TfidfVectorizer(max_features=100)
vectorizer.fit(data['message'])
joblib.dump(vectorizer, './vectorizer.pkl')  # Sauvegarder le vectoriseur

# Prétraiter un nouveau commit pour la prédiction
def preprocess_new_commit(commit_text):
    # Créer un dictionnaire par défaut avec des valeurs de remplacement pour toutes les caractéristiques
    new_commit = {
        'Date': pd.Timestamp.now(),
        'Duration': 0,  # Exemple : durée en secondes
        'User': 'UNKNOWN',
        'Author': 'UNKNOWN',
        'State': 'UNKNOWN',
        'Labels': 'UNKNOWN',
        'Year': pd.Timestamp.now().year,
        'Month': pd.Timestamp.now().month,
        'Day': pd.Timestamp.now().day
    }

    # Encoder les colonnes catégoriques en tenant compte des labels inconnus
    for col in categorical_columns:
        if new_commit[col] not in label_encoders[col].classes_:
            new_commit[col] = label_encoders[col].classes_[0]
        new_commit[col] = label_encoders[col].transform([new_commit[col]])[0]

    # Ajouter des caractéristiques TF-IDF du message de commit
    vectorizer = joblib.load('./vectorizer.pkl')
    tfidf_vector = vectorizer.transform([commit_text]).toarray()[0]
    for i, value in enumerate(tfidf_vector):
        new_commit[f'tfidf_feature_{i}'] = value

    # Ajouter les colonnes manquantes avec des valeurs par défaut
    missing_cols = set(column_names) - set(new_commit.keys())
    for col in missing_cols:
        new_commit[col] = 0

    # Réordonner les colonnes pour correspondre à l'ordre attendu
    new_commit_df = pd.DataFrame([new_commit], columns=column_names)

    # Normaliser les données
    scaler = joblib.load('./scaler.pkl')
    new_commit_scaled = scaler.transform(new_commit_df)

    print(f"Processed New Commit Data: {new_commit_df}")
    return new_commit_scaled

# Fonction pour prédire un nouveau commit
def predict_new_commit(commit_text, model_type='rf'):
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        model = joblib.load('./rf_model.pkl')
        new_commit_prediction_proba = model.predict_proba(new_commit_preprocessed)
    elif model_type == 'nn':
        model = load_model('./nn_model.keras')
        new_commit_prediction_proba = model.predict(new_commit_preprocessed)
    else:
        raise ValueError("Invalid model type specified. Choose 'rf' or 'nn'.")

    class_names = le_class.classes_
    predictions = {
        class_name: proba for class_name, proba in zip(class_names, new_commit_prediction_proba[0])
    }
    
    # Afficher les prédictions à la console
    print("\nPredictions:")
    for class_name, proba in predictions.items():
        print(f"{class_name}: {proba*100:.2f}%")

# Main execution
if __name__ == "__main__":
    commit_message = sys.argv[1] if len(sys.argv) > 1 else "No commit message provided"
    
    print(f"Commit Message: {commit_message}")
    
    print("Running Random Forest Prediction...")
    predict_new_commit(commit_message, model_type='rf')
    
    print("\nRunning Neural Network Prediction...")
    predict_new_commit(commit_message, model_type='nn')
