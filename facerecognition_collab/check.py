import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# --- CONFIG ---
ENCODINGS_FILE = "recognised/team_members/encodings.npz"
TEST_IMAGE_PATH = "sample3.jpeg"   # <-- change this to your sample image path
SIMILARITY_THRESHOLD = 0.35
# ---------------

# Load known encodings
if not os.path.exists(ENCODINGS_FILE):
    raise FileNotFoundError(f"❌ Encodings file not found: {ENCODINGS_FILE}")

data = np.load(ENCODINGS_FILE, allow_pickle=True)
known_encodings = data["encodings"]
photo_names = data.get("names", [f"user_{i}" for i in range(len(known_encodings))])
print(f"✅ Loaded {len(known_encodings)} known encodings from {ENCODINGS_FILE}")

# Initialize InsightFace model
print("🧠 Loading ArcFace model...")
face_app = FaceAnalysis(name="buffalo_l", root="/home/neub1t/.insightface")
face_app.prepare(ctx_id=0, det_size=(640, 640))
# face_app = FaceAnalysis(name="buffalo_l", root="/home/neub1t/.insightface")
# face_app.prepare(ctx_id=0, det_size=(320, 320), models=['detection', 'recognition'])

print("✅ Model ready!")

# Read the test image
if not os.path.exists(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"❌ Test image not found: {TEST_IMAGE_PATH}")

img = cv2.imread(TEST_IMAGE_PATH)
faces = face_app.get(img)

if len(faces) == 0:
    print("❌ No face detected in test image.")
    exit()

test_encoding = faces[0].embedding
test_norm = np.linalg.norm(test_encoding)

# Compute cosine similarity with all stored encodings
sims = np.dot(known_encodings, test_encoding) / (
    np.linalg.norm(known_encodings, axis=1) * test_norm
)

# Find the best match
best_index = int(np.argmax(sims))
best_score = float(sims[best_index])
matched_name = str(photo_names[best_index])

print("\n📊 --- Comparison Result ---")
print(f"Best Match: {matched_name}")
print(f"Similarity Score: {best_score:.4f}")

if best_score > SIMILARITY_THRESHOLD:
    print("✅ Face MATCHED (authorised)")
else:
    print("❌ No Match (unauthorised)")
