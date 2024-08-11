import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from termcolor import colored
import sys

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

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_sm, y_train_sm)

# Create and train the neural network model
y_train_sm_one_hot = to_categorical(y_train_sm)
y_test_one_hot = to_categorical(y_test)

nn_model = Sequential()
nn_model.add(Dense(128, input_dim=X_train_sm.shape[1], activation='relu'))
nn_model.add(Dropout(0.5))  # Add dropout to prevent overfitting
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(len(le_class.classes_), activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_sm, y_train_sm_one_hot, epochs=50, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Save the models and necessary objects
joblib.dump(rf_model, './rf_model.pkl')
joblib.dump(scaler, './scaler.pkl')
joblib.dump(label_encoders, './label_encoders.pkl')
joblib.dump(le_class, './label_encoder_class.pkl')  # Save the label encoder for classification

# Create and fit TF-IDF Vectorizer on commit messages
vectorizer = TfidfVectorizer(max_features=100)
vectorizer.fit(data['message'])

# Save the vectorizer
joblib.dump(vectorizer, './vectorizer.pkl')

print("Models and preprocessing objects saved successfully.")

# Preprocess a new commit for prediction
def preprocess_new_commit(commit_text):
    # Create a default dictionary with placeholder values for all features
    new_commit = {
        'Date': pd.Timestamp.now(),
        'Duration': 0,  # Example: duration in seconds
        'User': 'UNKNOWN',
        'Author': 'UNKNOWN',
        'State': 'UNKNOWN',
        'Labels': 'UNKNOWN',
        'Year': pd.Timestamp.now().year,
        'Month': pd.Timestamp.now().month,
        'Day': pd.Timestamp.now().day
    }

    # Encode categorical columns with handling for unknown labels
    for col in categorical_columns:
        if new_commit[col] not in label_encoders[col].classes_:
            new_commit[col] = label_encoders[col].classes_[0]
        new_commit[col] = label_encoders[col].transform([new_commit[col]])[0]

    # Add TF-IDF features from the commit message
    tfidf_vector = vectorizer.transform([commit_text]).toarray()[0]
    for i, value in enumerate(tfidf_vector):
        new_commit[f'tfidf_feature_{i}'] = value

    # Add any missing columns with default values
    missing_cols = set(column_names) - set(new_commit.keys())
    for col in missing_cols:
        new_commit[col] = 0

    # Reorder columns to match the expected order
    new_commit_df = pd.DataFrame([new_commit], columns=column_names)

    # Normalize the data
    new_commit_scaled = scaler.transform(new_commit_df)

    print(f"Processed New Commit Data: {new_commit_df}")
    return new_commit_scaled

# Function to predict a new commit
def predict_new_commit(commit_text, model_type='rf'):
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        model = joblib.load('./rf_model.pkl')
        new_commit_prediction_proba = model.predict_proba(new_commit_preprocessed)
    elif model_type == 'nn':
        model = load_model('./nn_model.keras')
        new_commit_prediction_proba = model.predict(new_commit_preprocessed)
    else:
        raise ValueError("Model type not supported")
    
    print("Raw Prediction Probabilities: ", new_commit_prediction_proba)
    
    decoded_classes = le_class.classes_
    prediction_proba = {decoded_classes[i]: new_commit_prediction_proba[0][i] * 100 for i in range(len(decoded_classes))}
    
    # Create formatted and colorful output, replacing class 0 with its actual meaning
    formatted_result = "\n".join([
        colored(f"{prob:.2f}% de probabilité que le commit soit classé comme {cls}", 'green')
        for cls, prob in prediction_proba.items()
    ])
    
    return formatted_result

# Read the commit message from command line arguments
if len(sys.argv) > 1:
    commit_message = sys.argv[1]
else:
    commit_message = "correction bug"  # Default message for testing

# Predict using the Random Forest model
rf_prediction = predict_new_commit(commit_message, model_type='rf')
print("\nPredictions (Random Forest):")
print(rf_prediction)

# Predict using the Neural Network model
nn_prediction = predict_new_commit(commit_message, model_type='nn')
print("\nPredictions (Neural Network):")
print(nn_prediction)
