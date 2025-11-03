import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ✅ 1. Load the extracted features
data = pd.read_csv(r"F:\SEMESTER 5\PROJECTS\MACHINE LEARNING PROJECT\ravdess_dataset.csv")

# Extract actor (speaker) ID from filename (Actor_01, Actor_02, etc.)
data['actor_id'] = data['file_name'].apply(lambda x: int(x.split('-')[-1].split('.')[0]) if '-' in x else int(x.split('_')[-1].split('.')[0]))

# Separate features and labels
X = data.drop(columns=['file_name', 'emotion_labels', 'actor_id']).values
y = data['emotion_labels'].values
actors = data['actor_id'].values

# Encode emotion labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 2. Improved MLP model structure (BatchNorm + Dropout + EarlyStopping)
def build_mlp(input_dim, output_dim):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ✅ 3. Leave-One-Speaker-Out Cross Validation
unique_actors = np.unique(actors)
results = []

for actor in unique_actors:
    print(f"\n🎭 Testing on Actor {actor} ...")

    # Split data: one actor for test, others for training
    train_idx = np.where(actors != actor)[0]
    test_idx = np.where(actors == actor)[0]

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_onehot[train_idx], y_onehot[test_idx]

    # Build and train model
    model = build_mlp(X_train.shape[1], y_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate on the left-out actor
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # ✅ Compute metrics
    acc = accuracy_score(y_true_classes, y_pred_classes)
    prec = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    rec = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

    print(f"Actor {actor} → Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")
    print("Confusion Matrix:")
    print(conf_mat)

    results.append([actor, acc, prec, rec, f1])

    # Save results for this actor
    actor_results = pd.DataFrame({
        'actor_id': [actor] * len(y_pred_classes),
        'true_label': y_true_classes,
        'predicted_label': y_pred_classes
    })
    actor_results.to_csv(f"actor_{actor}_predictions.csv", index=False)

# ✅ 4. Print overall LOSO performance
results_df = pd.DataFrame(results, columns=['Actor', 'Accuracy', 'Precision', 'Recall', 'F1'])
mean_metrics = results_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean()

print("\n====================================")
print("🎯 Average LOSO Performance")
print(f"Accuracy:  {mean_metrics['Accuracy']*100:.2f}%")
print(f"Precision: {mean_metrics['Precision']*100:.2f}%")
print(f"Recall:    {mean_metrics['Recall']*100:.2f}%")
print(f"F1 Score:  {mean_metrics['F1']*100:.2f}%")
print("====================================")

# Save overall summary
results_df.to_csv("overall_LOSO_results.csv", index=False)
