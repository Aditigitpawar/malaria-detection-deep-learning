import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model("malaria_cnn_model.h5")

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    processed_img = preprocess_image(filepath)
    prediction = model.predict(processed_img)

    confidence = float(prediction[0][0]) * 100

    if prediction[0][0] > 0.5:
        result = "Parasitized"
        prediction_class = "parasitized"
    else:
        result = "Uninfected"
        prediction_class = "uninfected"

    return render_template("index.html",
                           prediction=result,
                           confidence=round(confidence, 2),
                           image_path=filepath,
                           prediction_class=prediction_class)

if __name__ == "__main__":
    app.run(debug=True)