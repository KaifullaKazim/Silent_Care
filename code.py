import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack

# Load preprocessed dataset
df = pd.read_excel("D:\Downloads\Silent_Care\processed_therapeutic_solution.xlsx")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'previous_diagnosis', 'therapy_history', 'medication', 'diagnosis_condition']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoders for future use

# Encode target variable (suggested_therapy)
le_therapy = LabelEncoder()
df['suggested_therapy'] = le_therapy.fit_transform(df['suggested_therapy'])

# Use a faster BERT model for embeddings
bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
symptom_embeddings = np.array(bert_model.encode(df['symptoms'].tolist(), batch_size=32, show_progress_bar=True))

# Define features (X) and target (y)
numeric_features = df[['age', 'gender', 'duration_(weeks)', 'previous_diagnosis', 'therapy_history',
                        'medication', 'diagnosis_condition', 'mood', 'stress_level', 'urgency_level']]
X = np.hstack([symptom_embeddings, numeric_features])  # Combine BERT embeddings and numerical features
y = df['suggested_therapy']

# Handle class imbalance using SMOTE for multi-class classification
unique_classes, class_counts = np.unique(y, return_counts=True)
sampling_strategy = {class_label: int(count * 1.2) for class_label, count in zip(unique_classes, class_counts)}
smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning with a reduced GridSearchCV search space
param_grid = {
    'max_depth': [6],
    'learning_rate': [0.1],
    'n_estimators': [200],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le_therapy.classes_),
                              random_state=42, tree_method='hist', device='cuda')  # Fix GPU compatibility

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)  # Reduce folds to speed up

grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Optimized Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to predict therapy based on user input
def recommend_therapy(user_data):
    user_df = pd.DataFrame([user_data])
    for col in categorical_cols:
        if user_df[col][0] in label_encoders[col].classes_:
            user_df[col] = label_encoders[col].transform([user_df[col][0]])
        else:
            user_df[col] = -1  # Assign unseen categories to -1

    symptom_vector = bert_model.encode([user_df['symptoms'][0]])
    numeric_data = np.array(user_df[['age', 'gender', 'duration_(weeks)', 'previous_diagnosis', 'therapy_history',
                                     'medication', 'diagnosis_condition', 'mood', 'stress_level', 'urgency_level']])
    user_input_features = np.hstack([symptom_vector, numeric_data])

    therapy_pred = best_model.predict(user_input_features.reshape(1, -1))[0]
    return le_therapy.inverse_transform([therapy_pred])[0]

# Example usage
user_input = {'age': 30, 'gender': 'male', 'duration_(weeks)': 20, 'previous_diagnosis': 'anxiety',
              'therapy_history': 'yes', 'medication': 'no', 'diagnosis_condition': 'stress',
              'mood': 5, 'stress_level': 7, 'urgency_level': 2, 'symptoms': 'feeling anxious and trouble sleeping'}

# Convert categorical input to match encoding
for col in categorical_cols:
    if user_input[col] in label_encoders[col].classes_:
        user_input[col] = label_encoders[col].transform([user_input[col]])[0]
    else:
        user_input[col] = -1  # Assign unseen categories to -1

# Get therapy recommendation
recommended_therapy = recommend_therapy(user_input)
print("Recommended Therapy:", recommended_therapy)