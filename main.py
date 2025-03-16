import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import joblib

# Load preprocessed dataset
try:
    df = pd.read_csv("Threauptic_Solution.csv", encoding="utf-8")
except FileNotFoundError:
    print("Error: File 'Threauptic_Solution.csv' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading the dataset: {e}")
    exit()

# Define categorical columns
categorical_cols = ['Gender', 'Previous Diagnosis', 'Therapy History', 'Medication', 'Diagnosis / Condition']
label_encoders = {}

# Apply Label Encoding to categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoders for future use

# Encode target variable (Suggested Therapy)
le_therapy = LabelEncoder()
df['Suggested Therapy'] = le_therapy.fit_transform(df['Suggested Therapy'])

# 🔹 Fix: Convert Mood, Stress Level, and Urgency Level to numeric values
mood_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
stress_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
urgency_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}

df['Mood'] = df['Mood'].map(mood_mapping)
df['Stress Level'] = df['Stress Level'].map(stress_mapping)
df['Urgency Level'] = df['Urgency Level'].map(urgency_mapping)

# Ensure there are no NaN values after mapping
df.fillna(0, inplace=True)

# Load pre-trained BERT model for Symptoms encoding
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Convert Symptoms column to embeddings
symptom_embeddings = np.array(bert_model.encode(df['Symptoms'].astype(str).tolist(), batch_size=32, show_progress_bar=True))

# Define numerical features
numeric_features = df[['Age', 'Gender', 'Duration (weeks)', 'Previous Diagnosis', 
                       'Therapy History', 'Medication', 'Diagnosis / Condition', 
                       'Mood', 'Stress Level', 'Urgency Level']]

# Combine BERT embeddings with numerical features
X = np.hstack([symptom_embeddings, numeric_features])
y = df['Suggested Therapy']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'max_depth': [6],
    'learning_rate': [0.1],
    'n_estimators': [200],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le_therapy.classes_),
                              random_state=42, tree_method='hist', device='cuda')  # Use GPU if available

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Save the best model and encoders
joblib.dump(best_model, "therapy_recommendation_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(le_therapy, "therapy_encoder.pkl")

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Optimized Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_therapy.classes_))

# Function to predict therapy based on user input
def recommend_therapy(user_data):
    try:
        user_df = pd.DataFrame([user_data])
        
        # Encode categorical columns
        for col in categorical_cols:
            if user_df[col][0] in label_encoders[col].classes_:
                user_df[col] = label_encoders[col].transform([user_df[col][0]])
            else:
                # Dynamically add the new category to label encoder
                all_classes = np.append(label_encoders[col].classes_, user_df[col][0])
                label_encoders[col].classes_ = all_classes
                user_df[col] = label_encoders[col].transform([user_df[col][0]])


        # Encode Mood, Stress Level, and Urgency Level
        user_df['Mood'] = mood_mapping.get(user_df['Mood'][0], 2)  # Default to 'Moderate' if unknown
        user_df['Stress Level'] = stress_mapping.get(user_df['Stress Level'][0], 2)
        user_df['Urgency Level'] = urgency_mapping.get(user_df['Urgency Level'][0], 2)

        # Convert symptoms to BERT embeddings
        symptom_vector = bert_model.encode([user_df['Symptoms'][0]])
        
        # Extract numerical features
        numeric_data = np.array(user_df[['Age', 'Gender', 'Duration (weeks)', 'Previous Diagnosis', 
                                         'Therapy History', 'Medication', 'Diagnosis / Condition', 
                                         'Mood', 'Stress Level', 'Urgency Level']])
        
        # Combine numerical features and symptom embeddings
        user_input_features = np.hstack([symptom_vector, numeric_data])

        # Predict therapy
        therapy_pred = best_model.predict(user_input_features.reshape(1, -1))[0]
        return le_therapy.inverse_transform([therapy_pred])[0]
    except Exception as e:
        print(f"Error in therapy recommendation: {e}")
        return "Unable to recommend therapy"

# Example usage: Corrected user input dictionary
user_input = {
    'Age': 30,
    'Gender': 'male',
    'Duration (weeks)': 20,
    'Previous Diagnosis': 'anxiety',
    'Therapy History': 'yes',
    'Medication': 'no',
    'Diagnosis / Condition': 'stress',
    'Mood': 'Moderate',  # Can be 'Low', 'Moderate', or 'High'
    'Stress Level': 'High',
    'Urgency Level': 'Low',
    'Symptoms': 'feeling anxious and trouble sleeping'
}

# Get therapy recommendation
recommended_therapy = recommend_therapy(user_input)
print("Recommended Therapy:", recommended_therapy)
