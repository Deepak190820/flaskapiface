from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from collections import Counter

app = Flask(__name__)
CORS(app)  # 🔥 Enables CORS for all routes and origins

def bgr_to_hex(bgr):
    return '#{:02x}{:02x}{:02x}'.format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

def get_dominant_color(pixels):
    pixels = [tuple(np.round(p).astype(int)) for p in pixels if p is not None]
    color_counts = Counter(pixels)
    dominant = color_counts.most_common(1)[0][0]
    return dominant

@app.route("/")
def home():
    return "Skin Tone API is live!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 404

        x, y, w, h = faces[0]

        sample_regions = [
            (int(x + 0.3 * w), int(y + 0.4 * h)),
            (int(x + 0.7 * w), int(y + 0.4 * h)),
            (int(x + 0.5 * w), int(y + 0.2 * h)),
        ]

        skin_pixels = []
        for cx, cy in sample_regions:
            patch = img[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
            if patch.size > 0:
                skin_pixels.extend(patch.reshape(-1, 3))

        dominant_color = get_dominant_color(skin_pixels)
        hex_color = bgr_to_hex(dominant_color)

        return jsonify({
            "dominant_bgr": [int(c) for c in dominant_color],
            "dominant_hex": hex_color
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
