import os
import cv2
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "dataset"

classes = [
    "anorganik",
    "B3",
    "organik"
]

data = []
labels = []

print("Loading dataset...")

for label, folder in enumerate(classes):

    folder_path = os.path.join(DATASET_PATH, folder)

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        try:
            img = cv2.imread(file_path)

            if img is None:
                continue

            img = cv2.resize(img, (32, 32))
            img = img.flatten()

            data.append(img)
            labels.append(label)

        except:
            continue

X = np.array(data)
y = np.array(labels)

print(f"Total gambar : {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAkurasi:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("model_rf.pkl", "wb"))

print("\nModel berhasil disimpan sebagai model_rf.pkl")