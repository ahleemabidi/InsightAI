import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Chargement du dataset des commits
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
features = ['Year', 'Month', 'Day', 'Author', 'message', 'functions']  # À adapter selon votre dataset
target = 'Classification'  # La colonne que vous souhaitez prédire

# Séparation des données en ensembles d'entraînement et de test
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition des étapes de transformation
numeric_features = ['Year', 'Month', 'Day']
categorical_features = ['Author', 'message', 'functions']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # StandardScaler pour les features numériques
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncoder pour les features catégorielles
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Création du pipeline complet avec le modèle Random Forest Regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Calcul du MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Pourcentage de régression (exemple : normalisé entre 0 et 100)
# Si votre cible est binaire ou normalisée, vous devrez ajuster cela.
df_test = X_test.copy()
df_test['Actual'] = y_test.values
df_test['Predicted'] = y_pred

# Afficher le pourcentage de régression pour le dernier commit uniquement
last_commit_index = df_test.index[-1]
last_commit_row = df_test.loc[last_commit_index]

# Supposons que vous souhaitiez afficher la valeur prédite comme pourcentage
print(f"\nDernier Commit (Index {last_commit_index}):")
print(f"Pourcentage de régression après ce commit = {last_commit_row['Predicted']:.2f}%")

# Visualisation des valeurs prédites pour les commits
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df_test.index, y='Predicted', data=df_test, hue='Actual', palette='viridis', s=100)
plt.title('Valeurs Prédites pour chaque Commit (Random Forest Regressor)')
plt.xlabel('Index des Commits')
plt.ylabel('Valeur Prédite')
plt.legend(title='Valeur Réelle', loc='upper right', bbox_to_anchor=(1.3, 1))
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
