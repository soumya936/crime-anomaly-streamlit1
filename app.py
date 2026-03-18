from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

IMG_SIZE = (224, 224)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            # ✅ FIXED (correct indentation)
            file_path = "temp.jpg"
            file.save(file_path)

            img = load_img(file_path, target_size=IMG_SIZE)
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # 🔥 Dummy prediction (replace later)
            result = "🚨 ANOMALOUS"
            confidence = 87.5

    return render_template('index.html', result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=False)