
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

data_dir = "dataset/asl_alphabet_train"
X = []
y = []

# Load and preprocess
for label in os.listdir(data_dir):
    folder = os.path.join(data_dir, label)
    if not os.path.isdir(folder): continue
    for file in os.listdir(folder)[:200]:  # Limit to 200 per class to speed up
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save model
joblib.dump(knn, "knn_sign_model.joblib")
print("Model saved successfully.")
