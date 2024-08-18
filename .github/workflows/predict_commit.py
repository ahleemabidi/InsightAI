import pandas as pd
import numpy as np
import sys
import ast
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le fichier CSV
file_path = './.github/workflows/DATA_Finale.csv'
data = pd.read_csv(file_path)

# Prétraitement des données
data['Date'] = pd.to_datetime(data['Date'])
data['Duration'] = pd.to_timedelta(data['Duration']).dt.total_seconds()
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

data['Classification'] = data['Classification'].replace({
    'DOCUMENTATION (A NE PAS PRENDRE EN COMPTE)': 'OTHER',
    'BUG (PRIORITAIRE sur les autres LABELS)': 'BUG',
    'ÉVOLUTION (PRIORITAIRE sur les autres LABELS)': 'ÉVOLUTION'
})

# Encoder les colonnes catégorielles
categorical_columns = ['User', 'Author', 'State', 'Labels']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

le_class = LabelEncoder()
data['Classification'] = le_class.fit_transform(data['Classification'])

# TF-IDF Vectorizer pour la colonne 'message'
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_text = tfidf_vectorizer.fit_transform(data['message'])

# Préparer les données pour l'entraînement
X_numeric = data.drop(['Classification', 'Date', 'commit', 'message', 'functions', 'Created At', 'Updated At'], axis=1, errors='ignore')
X_numeric = StandardScaler().fit_transform(X_numeric)
X = np.hstack([X_numeric, X_text.toarray()])
y = data['Classification']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ajuster dynamiquement k_neighbors pour SMOTE
def get_k_neighbors(y_train):
    min_class_samples = min(y_train.value_counts())
    return min(max(1, min_class_samples - 1), 6)  # Ajuster k_neighbors avec un maximum de 6

k_neighbors = get_k_neighbors(y_train)

# Appliquer SMOTE
smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Vérifier la distribution des classes après SMOTE en pourcentage
class_distribution = pd.Series(y_train_sm).value_counts(normalize=True) * 100
print("Distribution des classes après SMOTE (en pourcentage) :")
print(class_distribution)

# Hyperparameter tuning pour Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}
rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
rf_grid.fit(X_train_sm, y_train_sm)

# Meilleur modèle Random Forest
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train_sm, y_train_sm)

# Évaluation du modèle Random Forest
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)

print("\nMatrice de confusion (Random Forest) :")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRapport de classification (Random Forest) :")
print(classification_report(y_test, y_pred_rf))

print(f"\nAccuracy Score (Random Forest) : {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print(f"ROC AUC Score (Random Forest) : {roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr') * 100:.2f}%")

# Créer et entraîner le modèle de réseau de neurones
y_train_sm_one_hot = to_categorical(y_train_sm)
y_test_one_hot = to_categorical(y_test)

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_sm.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(len(le_class.classes_), activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_sm, y_train_sm_one_hot, epochs=50, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Évaluation du modèle de réseau de neurones
y_pred_nn = nn_model.predict(X_test)
y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)

print("\nMatrice de confusion (Neural Network) :")
print(confusion_matrix(y_test, y_pred_nn_classes))

print("\nRapport de classification (Neural Network) :")
print(classification_report(y_test, y_pred_nn_classes))

print(f"\nAccuracy Score (Neural Network) : {accuracy_score(y_test, y_pred_nn_classes) * 100:.2f}%")
print(f"ROC AUC Score (Neural Network) : {roc_auc_score(y_test, y_pred_nn, multi_class='ovr') * 100:.2f}%")

# Mapper les indices aux noms de classes
class_index_mapping = {index: class_name for index, class_name in enumerate(le_class.classes_)}

# Fonction pour analyser l'impact d'un commit
def analyze_code_impact(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    return functions, classes

# Fonction pour calculer l'impact d'un commit
def calculate_impact(commit_diff):
    if commit_diff.strip() == "No previous commit to compare":
        print("Aucun précédent commit à comparer. Calcul de l'impact non possible.")
        return 0
    
    # Enregistrer le diff dans un fichier temporaire
    temp_diff_file = "temp_diff.txt"
    with open(temp_diff_file, "w") as file:
        file.write(commit_diff)
    
    # Analyser le diff
    affected_files = []  # Extraire les fichiers affectés du diff
    total_lines_changed = 0
    total_functions_affected = 0
    total_classes_affected = 0
    
    # Lire le fichier diff
    with open(temp_diff_file, "r") as file:
        for line in file:
            # Exemple de traitement, à adapter selon le format du diff
            if line.startswith("diff --git"):
                # Ajouter la logique pour extraire les fichiers affectés et les changements
                pass
    
    # Supprimer le fichier temporaire
    os.remove(temp_diff_file)
    
    impact_score = total_lines_changed + total_functions_affected + total_classes_affected
    return impact_score

# Prétraiter une nouvelle commit pour la prédiction
def preprocess_new_commit(commit_message, commit_diff):
    commit_message_tfidf = tfidf_vectorizer.transform([commit_message])
    new_commit_df = pd.DataFrame(commit_message_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    missing_cols = set(tfidf_vectorizer.get_feature_names_out()) - set(new_commit_df.columns)
    for col in missing_cols:
        new_commit_df[col] = 0
    new_commit_df = new_commit_df.fillna(0)
    
    # Calculer l'impact
    impact_score = calculate_impact(commit_diff)
    
    new_commit_df['impact_score'] = impact_score
    new_commit_preprocessed = StandardScaler().fit_transform(new_commit_df)
    
    return new_commit_preprocessed

# Fonction principale pour la prédiction d'un commit
def predict_new_commit(commit_message, commit_diff):
    preprocessed_commit = preprocess_new_commit(commit_message, commit_diff)
    
    # Prédiction avec le modèle Random Forest
    rf_prediction = rf_model.predict(preprocessed_commit)
    rf_probabilities = rf_model.predict_proba(preprocessed_commit)
    
    # Prédiction avec le modèle de réseau de neurones
    nn_prediction = np.argmax(nn_model.predict(preprocessed_commit), axis=1)
    
    # Mapper les indices des prédictions aux noms des classes
    rf_prediction_class = le_class.inverse_transform(rf_prediction)[0]
    nn_prediction_class = le_class.inverse_transform(nn_prediction)[0]
    
    print(f"\nPrédiction Random Forest : {rf_prediction_class}")
    print(f"Probabilités Random Forest : {rf_probabilities}")
    
    print(f"\nPrédiction Neural Network : {nn_prediction_class}")

# Principal
if __name__ == "__main__":
    commit_message = sys.argv[1]
    commit_diff = sys.argv[2]
    predict_new_commit(commit_message, commit_diff)
