import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score

# Charger les modèles
classification_pipeline = joblib.load('classification_pipeline.pkl')
regression_pipeline = joblib.load('regression_pipeline.pkl')

# Charger les données de test
file_path = file_path = './DATA_Finale.csv'

data = pd.read_csv(file_path)

# Convertir la colonne 'Date' en datetime
data['Date'] = pd.to_datetime(data['Date'])

# Transformer les dates en caractéristiques numériques
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Supprimer la colonne 'Date'
data = data.drop('Date', axis=1)

# Sélectionner les caractéristiques et les cibles
classification_features = ['Year', 'Month', 'Day', 'Author', 'message', 'functions']
classification_target = 'Classification'

X_classification = data[classification_features]
y_classification = data[classification_target]

# Prédictions avec le modèle de classification
y_pred_classification = classification_pipeline.predict(X_classification)

# Évaluation du modèle de classification
print("Classification Accuracy:", accuracy_score(y_classification, y_pred_classification))
print("\nClassification Report:\n", classification_report(y_classification, y_pred_classification, zero_division=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_classification, y_pred_classification))

# Identifier les fonctions avec des bugs
bug_indices = data.loc[y_pred_classification == 'BUG'].index
functions_with_bugs = data.loc[bug_indices, 'functions']
print("Fonctions avec des bugs après ce commit:", functions_with_bugs.tolist())
