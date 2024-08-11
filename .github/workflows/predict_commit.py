import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the CSV file
file_path = './.github/workflows/DATA_Finale.csv'
data = pd.read_csv(file_path)

# Convert date/time columns to datetime
data['Date'] = pd.to_datetime(data['Date'])
data['Created At'] = pd.to_datetime(data['Created At'])
data['Updated At'] = pd.to_datetime(data['Updated At'])

# Extract relevant features from date/time columns
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Convert duration to seconds
data['Duration'] = pd.to_timedelta(data['Duration']).dt.total_seconds()

# Merge rare classes
data['Classification'] = data['Classification'].replace({
    'DOCUMENTATION (A NE PAS PRENDRE EN COMPTE)': 'OTHER',
    'BUG (PRIORITAIRE sur les autres LABELS)': 'BUG',
    'ÉVOLUTION (PRIORITAIRE sur les autres LABELS)': 'ÉVOLUTION'
})

# Encode categorical columns
label_encoders = {}
categorical_columns = ['User', 'Author', 'State', 'Labels']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode the Classification column
le_class = LabelEncoder()
data['Classification'] = le_class.fit_transform(data['Classification'])

# Preprocess data
X = data.drop(['Classification', 'Date', 'commit', 'message', 'functions', 'Created At', 'Updated At'], axis=1, errors='ignore')
y = data['Classification']

# Save column names
column_names = list(X.columns)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dynamically adjust k_neighbors
def get_k_neighbors(y_train):
    min_class_samples = min(y_train.value_counts())
    return max(1, min_class_samples - 1)

k_neighbors = get_k_neighbors(y_train)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1)
rf_grid.fit(X_train_sm, y_train_sm)

# Best parameters and score
print("Best Parameters for Random Forest: ", rf_grid.best_params_)
print("Best Score for Random Forest: ", rf_grid.best_score_)

# Create and train the Random Forest model with the best parameters
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train_sm, y_train_sm)

# Create and train the neural network model with Dropout
y_train_sm_one_hot = to_categorical(y_train_sm)
y_test_one_hot = to_categorical(y_test)

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_sm.shape[1], activation='relu'))
nn_model.add(Dropout(0.5))  # Add Dropout layer
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dropout(0.5))  # Add another Dropout layer
nn_model.add(Dense(len(le_class.classes_), activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_sm, y_train_sm_one_hot, epochs=50, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Save the Keras model in the recommended format
nn_model.save('./nn_model.keras')

# Save models and necessary objects
joblib.dump(rf_model, './rf_model.pkl')
joblib.dump(scaler, './scaler.pkl')
joblib.dump(label_encoders, './label_encoders.pkl')
joblib.dump(le_class, './label_encoder_class.pkl')  # Save the label encoder for classification

# Save the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=100)
vectorizer.fit(data['message'])
joblib.dump(vectorizer, './vectorizer.pkl')
