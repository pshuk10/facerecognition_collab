from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import datetime
from insightface.app import FaceAnalysis

app = Flask(__name__)
user_path = "facerecognition_collab/recognised/team_members"

known_encodings = []
photo_names = []
photo_image_paths = []   # new: store an associated image path if available

# Initialize ArcFace model using the local buffalo_l model directory parent
print("Loading ArcFace model from local cache...")
face_app = FaceAnalysis(name="buffalo_l", root="/home/neub1t/.insightface")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ ArcFace model loaded successfully!")


def load_known_encodings():
    """Load saved .npy embeddings into memory and map to image files if present."""
    global known_encodings, photo_names, photo_image_paths
    known_encodings.clear()
    photo_names.clear()
    photo_image_paths.clear()

    print("Loading known face encodings from:", user_path)

    if not os.path.exists(user_path):
        print("User path does not exist:", user_path)
        return

    for file in sorted(os.listdir(user_path)):
        if file.endswith(".npy"):
            encoding_path = os.path.join(user_path, file)
            try:
                encoding = np.load(encoding_path)
                if encoding.ndim > 1:
                    encoding = encoding[0]
                known_encodings.append(encoding)
                photo_names.append(file)

                # try to find an associated image file in the same folder
                base = file
                # common patterns: timestamp_encoding.npy -> timestamp.jpg
                if base.endswith("_encoding.npy"):
                    base_name = base[:-len("_encoding.npy")]
                else:
                    base_name = os.path.splitext(base)[0]

                found_image = None
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    candidate = os.path.join(user_path, base_name + ext)
                    if os.path.exists(candidate):
                        found_image = candidate
                        break

                # if not found, also look for any jpg/png that starts with base_name
                if found_image is None:
                    for fn in os.listdir(user_path):
                        if fn.lower().endswith((".jpg", ".jpeg", ".png")) and fn.startswith(base_name):
                            found_image = os.path.join(user_path, fn)
                            break

                photo_image_paths.append(found_image)  # may be None
            except Exception as e:
                print(f"Failed to load {encoding_path}: {e}")

    print(f"Loaded {len(known_encodings)} known encodings.")


@app.route('/upload', methods=['POST'])
def recognise_face():
    """Receive image, compute ArcFace embedding, and compare."""
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # detect faces and get embedding using local buffalo model
    faces = face_app.get(img)

    if len(faces) == 0:
        return jsonify({"status": "no_face_detected"}), 400

    test_encoding = faces[0].embedding

    if not known_encodings:
        load_known_encodings()

    if not known_encodings:
        # still empty after loading
        print("No known encodings available to compare against.")
        return jsonify({"status": "unauthorised", "reason": "no_known_encodings"}), 200

    # compute cosine similarities efficiently (normalize vectors first)
    test_norm = np.linalg.norm(test_encoding)
    sims = []
    for known in known_encodings:
        k_norm = np.linalg.norm(known)
        if test_norm == 0 or k_norm == 0:
            sims.append(-1.0)
        else:
            sims.append(float(np.dot(test_encoding, known) / (test_norm * k_norm)))

    sims = np.array(sims, dtype=float)
    best_index = int(np.argmax(sims))
    best_score = float(sims[best_index])

    # threshold: tune between 0.3 - 0.5 for your dataset; left as 0.32 default
    if best_score > 0.32:
        matched_photo = photo_names[best_index]
        matched_image_path = photo_image_paths[best_index]  # may be None
        print(f"Match found: {matched_photo} (similarity={best_score:.3f}) image={matched_image_path}")
        # Return matched .npy and the associated image path (if any)
        return jsonify({
            "status": "authorised",
            "photo": matched_photo,
            "photo_image_path": matched_image_path
        }), 200
    else:
        print(f"No match (best similarity={best_score:.3f})")
        return jsonify({"status": "unauthorised", "best_score": best_score}), 200


@app.route('/reload', methods=['GET'])
def reload_encodings():
    load_known_encodings()
    return jsonify({"status": "reloaded", "total_encodings": len(known_encodings)}), 200


if __name__ == '__main__':
    load_known_encodings()
    app.run(host='0.0.0.0', port=8000)
