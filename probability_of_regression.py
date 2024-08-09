import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Chargement du dataset des commits (assurez-vous d'avoir préparé votre dataset selon les étapes nécessaires)
file_path = './DATA_Finale.csv'
data = pd.read_csv(file_path)

# Conversion de la colonne 'Date' en datetime si nécessaire
data['Date'] = pd.to_datetime(data['Date'])

# Transformation des dates en caractéristiques numériques
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Supprimer la colonne originale 'Date' si nécessaire
data = data.drop('Date', axis=1)

# Sélection des features et de la cible
features = ['Year', 'Month', 'Day', 'Author', 'message', 'functions']  # À adapter selon votre dataset
target = 'Classification'  # La colonne que vous souhaitez prédire (BUG, ÉVOLUTION, DOCUMENTATION, etc.)

# Séparation des données en ensembles d'entraînement et de test
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition des étapes de transformation
numeric_features = ['Year', 'Month', 'Day']
categorical_features = ['Author']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Utilisation de StandardScaler pour les features numériques
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Utiliser OneHotEncoder pour les features catégorielles
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Création du pipeline complet avec le modèle Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))  # Modèle de Random Forest avec 100 arbres
])

# Entraînement du modèle sur l'ensemble d'entraînement
pipeline.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Prédiction des probabilités de chaque classe
y_proba = pipeline.predict_proba(X_test)

# Probabilité de la classe "régression" (1 si vous utilisez un binaire 0/1)
regression_proba = y_proba[:, 1]  # Modifiez l'indice selon votre étiquette de régression

# Évaluation du modèle
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Imprimer la probabilité de régression pour chaque commit
for i, prob in enumerate(regression_proba):
    print(f"Commit {i+1}: Probabilité de régression = {prob:.2f}")

# Visualisation des probabilités de régression avec seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_test.index, y=regression_proba, hue=y_pred, palette='viridis', s=100)
plt.title('Probabilité de Régression pour chaque Commit (Random Forest)')
plt.xlabel('Index des Commits')
plt.ylabel('Probabilité de Régression')
plt.legend(title='Classification', loc='upper right', bbox_to_anchor=(1.3, 1))
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
