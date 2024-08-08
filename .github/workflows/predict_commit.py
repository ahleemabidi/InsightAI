import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import sys

# Custom category for replacing class 0
CUSTOM_CATEGORY = "Catégorie Personnalisée"

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
column_names = X.columns

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

# Make predictions and evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr')

print("Confusion Matrix (Random Forest):")
print(conf_matrix_rf)

print("\nClassification Report (Random Forest):")
print(class_report_rf)

print("\nAccuracy Score (Random Forest):")
print(accuracy_rf)

print("\nROC AUC Score (Random Forest):")
print(roc_auc_rf)

# Create and train the neural network model
y_train_sm_one_hot = to_categorical(y_train_sm)
y_test_one_hot = to_categorical(y_test)

nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_sm.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(len(le_class.classes_), activation='softmax'))

nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train_sm, y_train_sm_one_hot, epochs=50, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Save the Keras model
nn_model.save('./nn_model.h5')

# Make predictions and evaluate the neural network model
y_pred_nn = nn_model.predict(X_test)
y_pred_proba_nn = y_pred_nn

y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)

conf_matrix_nn = confusion_matrix(y_test, y_pred_nn_classes)
class_report_nn = classification_report(y_test, y_pred_nn_classes)
accuracy_nn = accuracy_score(y_test, y_pred_nn_classes)
roc_auc_nn = roc_auc_score(y_test, y_pred_proba_nn, multi_class='ovr')

print("Confusion Matrix (Neural Network):")
print(conf_matrix_nn)

print("\nClassification Report (Neural Network):")
print(class_report_nn)

print("\nAccuracy Score (Neural Network):")
print(accuracy_nn)

print("\nROC AUC Score (Neural Network):")
print(roc_auc_nn)

# Save models and necessary objects
joblib.dump(rf_model, './rf_model.pkl')
joblib.dump(scaler, './scaler.pkl')
joblib.dump(label_encoders, './label_encoders.pkl')
joblib.dump(le_class, './label_encoder_class.pkl')  # Save the label encoder for classification

# Preprocess a new commit for prediction
def preprocess_new_commit(commit_text):
    # Create a default dictionary with placeholder values for all features
    new_commit = {
        'Date': pd.Timestamp.now(),
        'Duration': 0,  # Example: duration in seconds
        'User': 0,
        'Author': 0,
        'State': 0,
        'Labels': 0,
        'Year': pd.Timestamp.now().year,
        'Month': pd.Timestamp.now().month,
        'Day': pd.Timestamp.now().day
    }

    # Encode categorical columns with handling for unknown labels
    for col in categorical_columns:
        if new_commit[col] not in label_encoders[col].classes_:
            # Print for debugging
            print(f"New commit value '{new_commit[col]}' for column '{col}' not found in encoder.")
            # Use the most frequent value as a fallback
            new_commit[col] = label_encoders[col].transform([0])[0]
        else:
            new_commit[col] = label_encoders[col].transform([new_commit[col]])[0]

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
    # Preprocess the new commit
    new_commit_preprocessed = preprocess_new_commit(commit_text)
    
    if model_type == 'rf':
        model = joblib.load('./rf_model.pkl')
        new_commit_prediction_proba = model.predict_proba(new_commit_preprocessed)
    elif model_type == 'nn':
        model = load_model('./nn_model.h5')
        new_commit_prediction_proba = model.predict(new_commit_preprocessed)
    else:
        raise ValueError("Model type not supported")
    
    # Print raw probabilities for debugging
    print("Raw Prediction Probabilities: ", new_commit_prediction_proba)
    
    # Decode classes
    decoded_classes = le_class.inverse_transform(np.arange(len(le_class.classes_)))
    
    # Map class names with the custom category
    class_mapping = {i: (CUSTOM_CATEGORY if i == 0 else decoded_classes[i]) for i in range(len(decoded_classes))}
    
    # Display the results
    result_strings = []
    for i, proba in enumerate(new_commit_prediction_proba[0]):
        class_name = class_mapping.get(i, "Unknown")
        result_strings.append(f"- **{class_name}**: {proba*100:.2f}%")

    # Format the final output
    formatted_result = (
        "### **Résultat de la Prédiction:**\n\n" +
        "\n".join(result_strings)
    )
    
    return formatted_result

# Read the commit message from command line arguments
if len(sys.argv) > 1:
    commit_message = sys.argv[1]
    model_type = 'rf'  # Default to Random Forest
    if len(sys.argv) > 2:
        model_type = sys.argv[2]
    # Make the prediction with the specified model
    result = predict_new_commit(commit_message, model_type=model_type)
    # Display the result with clear formatting
    print("\n====================\n")
    print(f"{result}")
    print("\n====================\n")
else:
    print("Veuillez fournir un message de commit en argument.")
