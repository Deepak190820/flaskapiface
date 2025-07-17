from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp
from collections import Counter

app = Flask(__name__)

def bgr_to_hex(bgr):
    return '#{:02x}{:02x}{:02x}'.format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

def get_dominant_color(pixels):
    pixels = [tuple(np.round(p).astype(int)) for p in pixels]
    color_counts = Counter(pixels)
    dominant = color_counts.most_common(1)[0][0]
    return dominant

mp_face_mesh = mp.solutions.face_mesh

def analyze_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

        for face_landmarks in results.multi_face_landmarks:
            indices = [205, 425, 151, 108, 200]
            skin_pixels = []

            for idx in indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                patch = img[max(0, y-5):y+5, max(0, x-5):x+5]
                skin_pixels.extend(patch.reshape(-1, 3))

            dominant_color = get_dominant_color(skin_pixels)
            hex_color = bgr_to_hex(dominant_color)
            return {"bgr": dominant_color, "hex": hex_color, "tone": "Wheatish"}

@app.route("/")
def home():
    return "Skin Tone API is live!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    base64_img = data.get("image")
    img_data = base64.b64decode(base64_img)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    result = analyze_face(img)

    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No face detected"}), 400
