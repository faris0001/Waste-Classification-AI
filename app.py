from flask import Flask, render_template, request
import pickle
import cv2
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open("model_rf.pkl", "rb"))

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = {
    0: "Anorganik ♻️",
    1: "B3 ☣️",
    2: "Organik 🌱"
}

@app.route("/")
def home():
    return render_template(
        "index.html",
        total_data=5626,
        accuracy="68.03%"
    )

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = cv2.imread(filepath)

    img = cv2.resize(img, (32,32))
    img = img.flatten()
    img = np.array(img).reshape(1,-1)

    prediction = model.predict(img)[0]

    probabilities = model.predict_proba(img)[0]

    confidence = round(
        np.max(probabilities) * 100,
        2
    )

    result = classes[prediction]

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        image=filepath,
        total_data=5626,
        accuracy="68.03%"
    )

if __name__ == "__main__":
    app.run(debug=True)