import pandas as pd
import numpy as np
import joblib  # Pour sauvegarder et charger les modèles
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

# Sauvegarder le TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Prétraiter les données
X_numeric = data.drop(['Classification', 'Date', 'commit', 'message', 'functions', 'Created At', 'Updated At'], axis=1, errors='ignore')
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X_numeric)  # Ajuster le StandardScaler sur les données d'entraînement
joblib.dump(scaler, 'scaler.pkl')  # Sauvegarder le StandardScaler

X = np.hstack([X_numeric, X_text.toarray()])
y = data['Classification']

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

# Sauvegarder le modèle Random Forest
joblib.dump(rf_model, 'rf_model.pkl')

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

# Sauvegarder le modèle Neural Network
nn_model.save('nn_model.h5')

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

# Mapper les indices aux noms de classes
class_index_mapping = {index: class_name for index, class_name in enumerate(le_class.classes_)}

# Prétraiter une nouvelle commit pour la prédiction
def preprocess_new_commit(commit_message):
    # Charger les objets de prétraitement
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Convertir le message du commit en vecteurs TF-IDF
    commit_message_tfidf = tfidf_vectorizer.transform([commit_message])

    # Créer un DataFrame avec les colonnes TF-IDF
    new_commit_df = pd.DataFrame(commit_message_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Ajouter les colonnes manquantes avec des valeurs par défaut
    missing_cols = set(tfidf_vectorizer.get_feature_names_out()) - set(new_commit_df.columns)
    for col in missing_cols:
        new_commit_df[col] = 0

    # Réordonner les colonnes selon l'ordre attendu
    all_columns = list(tfidf_vectorizer.get_feature_names_out())
    new_commit_df = new_commit_df[all_columns]

    # Prétraiter les données numériques
    new_commit_scaled = scaler.transform(new_commit_df)
    
    return new_commit_scaled

# Prédiction
def predict_commit(commit_message):
    new_commit_scaled = preprocess_new_commit(commit_message)
    
    # Charger les modèles
    rf_model = joblib.load('rf_model.pkl')
    nn_model = tf.keras.models.load_model('nn_model.h5')

    # Faire des prédictions avec les deux modèles
    rf_prediction = rf_model.predict(new_commit_scaled)
    nn_prediction = nn_model.predict(new_commit_scaled)
    
    rf_class = le_class.inverse_transform([np.argmax(rf_prediction)])
    nn_class = le_class.inverse_transform([np.argmax(nn_prediction)])
    
    return rf_class[0], nn_class[0]

# Exemple d'utilisation
commit_message = 'Fix bug in the payment module'
rf_class, nn_class = predict_commit(commit_message)

print(f"Prédiction (Random Forest) : {rf_class}")
print(f"Prédiction (Neural Network) : {nn_class}")
