import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Chargement du dataset
file_path = './DATA_Finale.csv'
data = pd.read_csv(file_path)

# Conversion de la colonne 'Date' en datetime
data['Date'] = pd.to_datetime(data['Date'])

# Transformation des dates en caractéristiques numériques
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Supprimer la colonne originale 'Date'
data = data.drop('Date', axis=1)

# Sélection des features et de la cible
features = ['Year', 'Month', 'Day', 'Author', 'message', 'functions']
target = 'Classification'

# Convertir les valeurs cibles en numériques
le = LabelEncoder()
data[target] = le.fit_transform(data[target])

# Vérification de la distribution des cibles
print("Distribution des cibles:")
print(data[target].value_counts())

# Séparation des données en ensembles d'entraînement et de test
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition des étapes de transformation
numeric_features = ['Year', 'Month', 'Day']
categorical_features = ['Author', 'message', 'functions']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Création du pipeline complet avec le modèle Random Forest Classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Évaluation du modèle
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Préparation des données pour affichage
df_test = X_test.copy()
df_test['Actual'] = y_test.values
df_test['Predicted'] = y_pred

# Afficher les valeurs prédites pour quelques commits
print("Exemples de prédictions:")
print(df_test[['Predicted', 'Actual']].head(20))

# Visualisation des données
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de Confusion (Random Forest Classifier)')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.show()

# Afficher le pourcentage de régression pour le dernier commit uniquement
last_commit_index = df_test.index[-1]
last_commit_row = df_test.loc[last_commit_index]

print(f"\nDernier Commit (Index {last_commit_index}):")
print(f"Pourcentage de régression après ce commit = {last_commit_row['Predicted']:.2f}%")
