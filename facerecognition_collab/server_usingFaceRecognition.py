from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import face_recognition
import datetime

app = Flask(__name__)
user_path = "facerecognition_collab/recognised/team_members"

known_encodings = []
photo_names = []   # 👈 Added to track which photo each encoding belongs to


def load_known_encodings():
    """Load all saved .npy encodings into memory once."""
    global known_encodings, photo_names
    known_encodings.clear()
    photo_names.clear()

    print("Loading known face encodings...")

    for file in os.listdir(user_path):
        if file.endswith(".npy"):
            encoding_path = os.path.join(user_path, file)
            try:
                encoding = np.load(encoding_path)
                known_encodings.append(encoding)
                photo_names.append(file)   # 👈 Store photo file name
            except Exception as e:
                print(f"Failed to load {encoding_path}: {e}")

    print(f"Loaded {len(known_encodings)} known encodings.")


@app.route('/upload', methods=['POST'])
def recognise_face():
    """Receive an image, create encoding, and compare with cached known encodings."""
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    if len(face_encodings) == 0:
        return jsonify({"status": "no_face_detected"}), 400

    test_encoding = face_encodings[0]

    if not known_encodings:
        load_known_encodings()

    results = face_recognition.compare_faces(known_encodings, test_encoding, tolerance=0.45)
    distances = face_recognition.face_distance(known_encodings, test_encoding)

    if True in results:
        best_index = np.argmin(distances)
        matched_photo = photo_names[best_index]  # 👈 Get matching photo name
        print(f"Match found: {matched_photo} (distance={distances[best_index]:.3f})")
        return jsonify({"status": "authorised", "photo": matched_photo}), 200
    else:
        print("No match found.")
        return jsonify({"status": "unauthorised"}), 200


@app.route('/reload', methods=['GET'])
def reload_encodings():
    """Reload known encodings manually without restarting server."""
    load_known_encodings()
    return jsonify({"status": "reloaded", "total_encodings": len(known_encodings)}), 200


if __name__ == '__main__':
    load_known_encodings()
    app.run(host='0.0.0.0', port=8000)


