import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Paths
dataset_path = "asl_alphabet_train"  
output_csv = "landmark_dataset.csv"
model_filename = "sign_model.pkl"

# Initialize mediapipe hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

data = []
labels = []

# Go through each label folder
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    print(f"Processing label: {label}")
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmark_points = []
                for lm in hand_landmarks.landmark:
                    landmark_points.extend([lm.x, lm.y, lm.z])
                if len(landmark_points) == 63:
                    data.append(landmark_points)
                    labels.append(label)

# Save to CSV (optional)
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(output_csv, index=False, header=False)
print(f"Saved structured landmark data to {output_csv}")

# Train model
X = np.array(data)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save model
joblib.dump(knn, model_filename)
print(f"Trained model saved to {model_filename}")

