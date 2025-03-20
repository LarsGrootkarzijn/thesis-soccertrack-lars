import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ----------------------
# CONFIG
# ----------------------
DATA_PATH = './combined_ball_data.csv'  # Your dataset path
EVENT_COLUMN = 'event'
RANDOM_STATE = 42
SAVE_DIR = './models_matrix/'

# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------
# LOAD DATA
# ----------------------
print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# ----------------------
# FEATURE SELECTION
# ----------------------
selected_features = ['acceleration']  # Only use 'acceleration' as input feature

# Verify columns exist
for col in selected_features:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset!")

X = df[selected_features]
y = df[EVENT_COLUMN]

# ----------------------
# BALANCED RESAMPLING WHILE PRESERVING ACCELERATION CHARACTERISTICS
# ----------------------

# Separate majority and minority classes
class_0 = df[df[EVENT_COLUMN] == 0]
class_1 = df[df[EVENT_COLUMN] == 1]

# Calculate the desired size for each class (e.g., make them both the same size)
n_samples = min(len(class_0), len(class_1))

# Resample class 0 and class 1 to have the same number of samples while preserving acceleration distribution
class_0_resampled = class_0.sample(n=n_samples, random_state=RANDOM_STATE, replace=False)
class_1_resampled = class_1.sample(n=n_samples, random_state=RANDOM_STATE, replace=False)

# Combine the resampled classes
df_balanced = pd.concat([class_0_resampled, class_1_resampled])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Print class distribution after resampling
print("\nBalanced class distribution:")
print(df_balanced[EVENT_COLUMN].value_counts())

# ----------------------
# SAVE THE BALANCED DATASET CSV
# ----------------------
balanced_csv_path = os.path.join(SAVE_DIR, 'balanced_dataset.csv')
df_balanced.to_csv(balanced_csv_path, index=False)
print(f"\nBalanced dataset saved to: {balanced_csv_path}")

# ----------------------
# PREPARE FEATURES AND LABELS
# ----------------------
X_balanced = df_balanced[selected_features].values  # Only acceleration as feature
y_balanced = df_balanced[EVENT_COLUMN].values

# ----------------------
# TRAIN-TEST SPLIT
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=RANDOM_STATE, stratify=y_balanced
)

# ----------------------
# NORMALIZE FEATURES
# ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# NEURAL NETWORK WITH 5 LAYERS
# ----------------------

model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),  # First hidden layer
    Dense(64, activation='relu'),  # Second hidden layer
    Dense(32, activation='relu'),  # Third hidden layer
    Dense(16, activation='relu'),  # Fourth hidden layer
    Dense(8, activation='relu'),   # Fifth hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adjusted learning rate
    loss='binary_crossentropy',  # Binary cross-entropy for binary classification
    metrics=['accuracy']
)

# ----------------------
# TRAIN THE MODEL
# ----------------------
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=200,  # Set a reasonable number of epochs
    batch_size=32,  # Set a larger batch size
    verbose=1
)

# ----------------------
# EVALUATION
# ----------------------
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

# ----------------------
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ----------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save confusion matrix image
matrix_path = os.path.join(SAVE_DIR, 'confusion_matrix_nn_balanced.png')
plt.savefig(matrix_path)
plt.show()

print(f"\nConfusion matrix saved to: {matrix_path}")
