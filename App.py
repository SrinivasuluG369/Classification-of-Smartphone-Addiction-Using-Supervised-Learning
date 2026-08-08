import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

print("--- ALL LIBRARIES LOADED SUCCESSFULLY ---")
print("="*70)

# ==============================================================================
# --- 2. DATA LOADING ---
# ==============================================================================
FILE_NAME = 'teen_phone_addiction_dataset.csv'

try:
    df = pd.read_csv(FILE_NAME)
    print(f"Dataset Loaded Successfully! Shape: {df.shape}")
    print(df.head(3))
except FileNotFoundError:
    print("ERROR: File not found. Please upload the dataset correctly.")
    exit()

print("="*70)

# ==============================================================================
# --- 3. CREATE BINARY TARGET COLUMN ---
# ==============================================================================
TARGET_COLUMN = 'Addiction_Level'
ADDICTION_THRESHOLD = 8.0

df['Is_Addicted'] = (df[TARGET_COLUMN] >= ADDICTION_THRESHOLD).astype(int)

print("Binary Target Column 'Is_Addicted' created successfully!")
print(df['Is_Addicted'].value_counts())
print("="*70)

# ==============================================================================
# --- 4. DEFINE FEATURES (X) AND TARGET (Y) ---
# ==============================================================================
TARGET_COLUMN_BINARY = 'Is_Addicted'

cols_to_drop = ['ID', 'Name', 'Location', TARGET_COLUMN, TARGET_COLUMN_BINARY]
X = df.drop(cols_to_drop, axis=1, errors='ignore')
Y = df[TARGET_COLUMN_BINARY]

print(f"Total Features Selected: {X.shape[1]}")
print("Target Distribution:")
print(Y.value_counts())
print("="*70)

# ==============================================================================
# --- 5. DEFINE NUMERICAL & CATEGORICAL FEATURE LISTS ---
# ==============================================================================
numerical_features = [
    'Age', 'Daily_Usage_Hours', 'Sleep_Hours', 'Anxiety_Level',
    'Depression_Level', 'Self_Esteem', 'Screen_Time_Before_Bed',
    'Phone_Checks_Per_Day', 'Apps_Used_Daily', 'Time_on_Social_Media',
    'Time_on_Gaming', 'Time_on_Education', 'Family_Communication',
    'Weekend_Usage_Hours'
]

categorical_features = [
    'Gender', 'School_Grade', 'Parental_Control', 'Phone_Usage_Purpose'
]

print("Numerical & Categorical Features Identified.")
print("="*70)

# ==============================================================================
# --- 6. PREPROCESSING PIPELINES ---
# ==============================================================================
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
], remainder='passthrough')

print("Preprocessing Pipeline Ready.")
print("="*70)

# ==============================================================================
# --- 7. TRAIN-TEST SPLIT ---
# ==============================================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    stratify=Y
)

print("Train-Test Split Completed.")
print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
print("="*70)

# ==============================================================================
# --- 8. APPLY PREPROCESSING ---
# ==============================================================================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing Applied Successfully!")
print(f"Processed Training Shape: {X_train_processed.shape}")
print(f"Processed Testing Shape: {X_test_processed.shape}")
print("="*70)

# ==============================================================================
# --- 9. MODEL TRAINING & TUNING ---
# ==============================================================================
SCORING_METRIC = 'recall'
print("--- Phase 3: Model Tuning Started ---")

log_reg = LogisticRegression(max_iter=1000, random_state=42)
grid_lr = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}

svc = SVC(probability=True, random_state=42, class_weight='balanced')
grid_svc = {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 0.1, 1]}

rand_forest = RandomForestClassifier(random_state=42)
grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [5, 10]}

models = [
    ('Logistic Regression', log_reg, grid_lr),
    ('Support Vector Machine', svc, grid_svc),
    ('Random Forest', rand_forest, grid_rf)
]

tuned_models = {}

for name, model, params in models:
    gs = GridSearchCV(model, params, cv=5, scoring=SCORING_METRIC, n_jobs=-1)
    gs.fit(X_train_processed, Y_train)
    tuned_models[name] = gs.best_estimator_
    print(f"{name} Best Recall (CV): {gs.best_score_:.4f}")

print("="*70)

# ==============================================================================
# --- 10. FINAL EVALUATION (PHASE 4) ---
# ==============================================================================
print("\n--- PHASE 4: FINAL MODEL EVALUATION ---")

results = {}

for name, model in tuned_models.items():
    Y_pred = model.predict(X_test_processed)

    report = classification_report(
        Y_test,
        Y_pred,
        output_dict=True,
        zero_division=0
    )

    results[name] = {
        "Accuracy": report['accuracy'],
        "Precision (Addicted)": report['1']['precision'],
        "Recall (Addicted)": report['1']['recall'],
        "F1 Score (Addicted)": report['1']['f1-score']
    }

    print(f"\nModel: {name}")
    print(classification_report(Y_test, Y_pred, zero_division=0))

results_df = pd.DataFrame(results).T
best_model = results_df['Recall (Addicted)'].idxmax()

print("\n--- FINAL MODEL COMPARISON ---")
print(results_df.sort_values(by='Recall (Addicted)', ascending=False))
print("="*70)

print(f"BEST MODEL BASED ON RECALL: {best_model}")
print("=== FULL PIPELINE EXECUTED SUCCESSFULLY ===")
