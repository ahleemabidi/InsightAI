import sys
import pandas as pd
import numpy as np
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

# Initialiser le TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_text = tfidf_vectorizer.fit_transform(data['message'])

# Prétraiter les données
X_numeric = data.drop(['Classification', 'Date', 'commit', 'message', 'functions', 'Created At', 'Updated At'], axis=1, errors='ignore')
X_numeric = StandardScaler().fit_transform(X_numeric)
X = np.hstack([X_numeric, X_text.toarray()])
y = data['Classification']

# Sauvegarder les noms de colonnes pour les données numériques et TF-IDF
numeric_column_names = X_numeric.shape[1]
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ajuster dynamiquement k_neighbors pour SMOTE
def get_k_neighbors(y_train):
    min_class_samples = min(y_train.value_counts())
    return max(1, min_class_samples - 1)

k_neighbors = get_k_neighbors(y_train)

# Appliquer SMOTE pour équilibrer les classes sur l'ensemble d'entraînement
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

# Meilleurs paramètres et score
print("Meilleurs paramètres pour Random Forest : ", rf_grid.best_params_)
print("Meilleur score pour Random Forest : ", rf_grid.best_score_)

# Créer et entraîner le modèle Random Forest avec les meilleurs paramètres
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train_sm, y_train_sm)

# Faire des prédictions et évaluer le modèle Random Forest
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr')

print("\nMatrice de confusion (Random Forest) :")
print(conf_matrix_rf)

print("\nRapport de classification (Random Forest) :")
for label, metrics in class_report_rf.items():
    if label == 'accuracy':
        continue
    print(f"Classe {label} :")
    print(f"  Précision : {metrics['precision'] * 100:.2f}%")
    print(f"  Rappel : {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score : {metrics['f1-score'] * 100:.2f}%")

print(f"\nAccuracy Score (Random Forest) : {accuracy_rf * 100:.2f}%")
print(f"ROC AUC Score (Random Forest) : {roc_auc_rf * 100:.2f}%")

# Créer et entraîner le modèle de réseau de neurones
y_train_sm_one_hot = to_categorical(y_train_sm)
y_test_one_hot = to_categorical(y_test)

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_sm.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(len(le_class.classes_), activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_sm, y_train_sm_one_hot, epochs=50, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Faire des prédictions et évaluer le modèle de réseau de neurones
y_pred_nn = nn_model.predict(X_test)
y_pred_proba_nn = y_pred_nn

y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)

conf_matrix_nn = confusion_matrix(y_test, y_pred_nn_classes)
class_report_nn = classification_report(y_test, y_pred_nn_classes, output_dict=True)
accuracy_nn = accuracy_score(y_test, y_pred_nn_classes)
roc_auc_nn = roc_auc_score(y_test, y_pred_proba_nn, multi_class='ovr')

print("\nMatrice de confusion (Neural Network) :")
print(conf_matrix_nn)

print("\nRapport de classification (Neural Network) :")
for label, metrics in class_report_nn.items():
    if label == 'accuracy':
        continue
    print(f"Classe {label} :")
    print(f"  Précision : {metrics['precision'] * 100:.2f}%")
    print(f"  Rappel : {metrics['recall'] * 100:.2f}%")
    print(f"  F1-score : {metrics['f1-score'] * 100:.2f}%")

print(f"\nAccuracy Score (Neural Network) : {accuracy_nn * 100:.2f}%")
print(f"ROC AUC Score (Neural Network) : {roc_auc_nn * 100:.2f}%")

# Prétraiter une nouvelle commit pour la prédiction
def preprocess_new_commit(commit_message):
    # Convertir le message du commit en vecteurs TF-IDF
    commit_message_tfidf = tfidf_vectorizer.transform([commit_message])

    # Créer un DataFrame avec les colonnes TF-IDF
    new_commit_df = pd.DataFrame(commit_message_tfidf.toarray(), columns=tfidf_feature_names)

    # Ajouter les colonnes manquantes avec des valeurs par défaut
    missing_cols = set(tfidf_feature_names) - set(new_commit_df.columns)
    for col in missing_cols:
        new_commit_df[col] = 0

    # Réordonner les colonnes selon l'ordre attendu
    all_columns = list(tfidf_feature_names)
    new_commit_df = new_commit_df[all_columns]

    # Normaliser les données
    scaler = StandardScaler()
    new_commit_numeric = np.zeros((1, numeric_column_names))
    new_commit_df = np.hstack([new_commit_numeric, new_commit_df])

    return new_commit_df

# Prédiction avec les modèles
def predict_new_commit(commit_message):
    new_commit_preprocessed = preprocess_new_commit(commit_message)

    # Faire la prédiction avec les deux modèles
    new_commit_prediction_proba_rf = rf_model.predict_proba(new_commit_preprocessed)
    new_commit_prediction_proba_nn = nn_model.predict(new_commit_preprocessed)

    # Résultats Random Forest
    print("\nPrediction Random Forest :")
    for i, class_name in enumerate(le_class.classes_):
        print(f"{class_name} : {new_commit_prediction_proba_rf[0][i] * 100:.2f}%")

    # Résultats Neural Network
    print("\nPrediction Neural Network :")
    for i, class_name in enumerate(le_class.classes_):
        print(f"{class_name} : {new_commit_prediction_proba_nn[0][i] * 100:.2f}%")

# Lecture du message du commit depuis les arguments
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erreur : Aucun message de commit fourni.")
        sys.exit(1)

    commit_message = sys.argv[1]
    predict_new_commit(commit_message)
