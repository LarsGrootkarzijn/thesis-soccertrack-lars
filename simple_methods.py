import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------
# CONFIG
# ----------------------
DATA_PATH = './combined_ball_data.csv'  # Path to your CSV
EVENT_COLUMN = 'event'
RANDOM_STATE = 42

# ----------------------
# LOAD DATA
# ----------------------
print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# ----------------------
# FEATURE SELECTION - Only velocity, acceleration, angle
# ----------------------
selected_features = ['velocity', 'acceleration', 'angle']

# Verify columns exist
for col in selected_features:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset!")

print(f"Using features: {selected_features}")

# ----------------------
# DOWNSAMPLE MAJORITY CLASS TO MINORITY CLASS SIZE
# ----------------------
class_0 = df[df[EVENT_COLUMN] == 0]
class_1 = df[df[EVENT_COLUMN] == 1]

print(f"Original class counts: 0 -> {len(class_0)}, 1 -> {len(class_1)}")

if len(class_0) > len(class_1):
    class_0_downsampled = class_0.sample(n=len(class_1), random_state=RANDOM_STATE)
    df_balanced = pd.concat([class_0_downsampled, class_1])
else:
    class_1_downsampled = class_1.sample(n=len(class_0), random_state=RANDOM_STATE)
    df_balanced = pd.concat([class_0, class_1_downsampled])

df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# ----------------------
# HANDLE NaN VALUES
# ----------------------
print("Checking for NaNs after balancing...")
print(df_balanced[selected_features].isna().sum())

# Drop rows with NaNs
df_balanced = df_balanced.dropna(subset=selected_features)

# Alternatively, you can fill NaNs like this:
# df_balanced[selected_features] = df_balanced[selected_features].fillna(0)

print("Balanced dataset class counts:", df_balanced[EVENT_COLUMN].value_counts().to_dict())

# ----------------------
# PREPARE FEATURES AND LABELS
# ----------------------
X = df_balanced[selected_features].values
y = df_balanced[EVENT_COLUMN].values

# ----------------------
# TRAIN-TEST SPLIT
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ----------------------
# NORMALIZE FEATURES
# ----------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# MODEL 1: RANDOM FOREST
# ----------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

# ----------------------
# MODEL 2: LOGISTIC REGRESSION
# ----------------------
lr_model = LogisticRegression(random_state=RANDOM_STATE)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

# ----------------------
# MODEL 3: K-NEAREST NEIGHBORS
# ----------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)

# ----------------------
# EVALUATE ALL MODELS
# ----------------------
def evaluate_model(name, y_test, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Random Forest Results
evaluate_model("Random Forest", y_test, rf_preds)

# Logistic Regression Results
evaluate_model("Logistic Regression", y_test, lr_preds)

# KNN Results
evaluate_model("K-Nearest Neighbors", y_test, knn_preds)
