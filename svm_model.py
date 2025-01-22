import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
print("Loading, finished importing modules...")

# File paths for model and scaler
model_path = 'best_svm_model.joblib'
scaler_path = 'scaler.joblib'

# Function to train and save the model


def train_and_save_model():
    print("Existing model not present, training and saving model.")
    print("Loading, starting data extraction from spreadsheet...")
    # Load and preprocess the data
    file_path = 'Python_Data.xlsx'

    # Read let-7b data from the file
    let7b_data = pd.ExcelFile(file_path).parse('hsa-let-7b')

    # Combine datasets and clean data
    columns_needed = ['Stage', 'read_per_million_miRNAMapped']
    let7b_combined = let7b_data[columns_needed].copy()

    # Encode target variable for binary classification
    let7b_combined['Cancer_Label'] = (let7b_combined['Stage'] == 'Cancer').astype(int)

    # If the target variable does not have both classes, raise an error
    if let7b_combined['Cancer_Label'].nunique() < 2:
        raise ValueError("The target variable must have at least two classes (e.g., Normal and Cancer).")

    # Separate features and target
    X = let7b_combined[['read_per_million_miRNAMapped']]
    y = let7b_combined['Cancer_Label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Loading, applying SMOTE and training model...")
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Split the resampled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train SVM model using GridSearchCV
    svm = SVC(probability=True, random_state=42)
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    svm_grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    svm_grid.fit(X_train, y_train)

    print("Loading, best model getting evaluated...\n\n")
    # Best model evaluation
    best_model = svm_grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    # Model information clculation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # Model info stored
    model_info_txt = f"""Accuracy: {accuracy}\n
Precision: {precision}\n
Recall: {recall}\n
ROC-AUC: {roc_auc}\n
Confusion Matrix:\n{conf_matrix}\n
Classification Report:\n{report}"""
    # Print results and save to a txt file
    print(model_info_txt)
    model_info_file = open("model_info", "w+")
    model_info_file.write(model_info_txt)
    model_info_file.close()
    # Save the model and scaler
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    # If model doesn't already exist, then create it
    train_and_save_model()
else:
    # Model already exists, proceed with extraction from joblib and use the existing model
    print(f"Model and scaler already exist at {model_path} and {scaler_path}.")
# Load the model and scaler
best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
# Displays the info on the model if the user wishes so
model_info = input("Do you want the info on the model loaded? Y for Yes, N for No\n")
model_info_result = True if model_info == "Y" else False
if model_info_result:
    model_info_read = open("model_info", "r+")
    print(model_info_read.read())


def rpm_identification():
    # Predict binary cancer status for user-input RPM
    rpm = float(input("What is the given RPM for let-7b?\n"))
    # Convert input to DataFrame with the correct feature name to avoid warning
    rpm_df = pd.DataFrame([[rpm]], columns=['read_per_million_miRNAMapped'])
    rpm_scaled = scaler.transform(rpm_df)
    # Predict with rpm and give probability
    cancer_prediction = best_model.predict(rpm_scaled)
    cancer_prob = best_model.predict_proba(rpm_scaled)[0][1]
    normal_prob = 1 - cancer_prob
    # Display results
    if cancer_prediction[0] == 1:
        print(f"Predicted: Cancer, {cancer_prob * 100:.1f}% chance")
    else:
        print(f"Predicted: Normal, {normal_prob * 100:.1f}% chance")
    # Ask for repeat
    repeat_input = input("Do you want to test again? Y for Yes, N for No.\n")
    repeat = True if repeat_input == "Y" else False
    if repeat:
        rpm_identification()
# takes user input and classifies whether it is Normal or Cancer RPM, using the model
rpm_identification()
