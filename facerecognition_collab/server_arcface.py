from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis



app = Flask(__name__)

# --- CONFIG ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../facerecognition_collab"))
ENCODINGS_FILE = os.path.join(BASE_DIR, "recognised/team_members/encodings.npz")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODEL_ROOT = "/home/neub1t/.insightface"
SIMILARITY_THRESHOLD = 0.32

# --- Ensure images folder exists ---
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Global variables ---
known_encodings = None
photo_names = None

# --- Load ArcFace model ---
print("LOADING ARCFACE MODEL FROM BUFFALO_L...")
face_app = FaceAnalysis(name="buffalo_l", root=MODEL_ROOT)
face_app.prepare(ctx_id=0, det_size=(480, 480))
print("ArcFace model loaded successfully!\n")


def load_known_encodings():
    """Load all known encodings and corresponding names from .npz file."""
    global known_encodings, photo_names

    if not os.path.exists(ENCODINGS_FILE):
        print(f"Encodings file not found: {ENCODINGS_FILE}")
        known_encodings, photo_names = np.array([]), []
        return

    data = np.load(ENCODINGS_FILE, allow_pickle=True)
    known_encodings = data["encodings"]
    photo_names = data.get("names", [f"user_{i}" for i in range(len(known_encodings))])
    print(f"Loaded {len(known_encodings)} known encodings from {ENCODINGS_FILE}")


@app.route('/upload', methods=['POST'])
def recognise_face():
    """Receive image, save it, compute ArcFace embedding, and compare."""
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    # img = cv2.rotate(img, cv2.ROTATE_180)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # --- Save uploaded image ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(IMAGES_DIR, f"upload_{timestamp}.jpg")
    cv2.imwrite(save_path, img)
    print(f"Saved incoming image at: {save_path}")

    # Detect faces
    faces = face_app.get(img)
    if len(faces) == 0:
        return jsonify({"status": "no_face_detected"}), 400

    test_encoding = faces[0].embedding

    # Load encodings if not already loaded
    if known_encodings is None or len(known_encodings) == 0:
        load_known_encodings()

    if len(known_encodings) == 0:
        return jsonify({"status": "unauthorised", "reason": "no_known_encodings"}), 200

    # Compute cosine similarities
    test_norm = np.linalg.norm(test_encoding)
    known_norms = np.linalg.norm(known_encodings, axis=1)
    sims = np.dot(known_encodings, test_encoding) / (known_norms * test_norm)

    best_index = int(np.argmax(sims))
    best_score = float(sims[best_index])
    matched_name = str(photo_names[best_index])

    # Decision
    if best_score > SIMILARITY_THRESHOLD:
        print(f"Match found: {matched_name} (similarity={best_score:.3f})")
        return jsonify({
            "status": "authorised",
            "photo": matched_name,
            "similarity": best_score,
            "saved_image": save_path
        }), 200
    else:
        print(f"No match (best similarity={best_score:.3f})")
        return jsonify({
            "status": "unauthorised",
            "best_score": best_score,
            "saved_image": save_path
        }), 200


@app.route('/reload', methods=['GET'])
def reload_encodings():
    """Reload all encodings manually."""
    load_known_encodings()
    return jsonify({"status": "reloaded", "total_encodings": len(known_encodings)}), 200


if __name__ == '__main__':
    load_known_encodings()
    app.run(host='0.0.0.0', port=8000)
