# import cv2
# import os
# import numpy as np

# # Load OpenCV’s built-in Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # --- STEP 1: Load and detect faces in two images ---
# def detect_face(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     if len(faces) == 0:
#         print(f"No face found in {image_path}")
#         return None

#     (x, y, w, h) = faces[0]  # Take first detected face
#     face_crop = gray[y:y+h, x:x+w]
#     face_resized = cv2.resize(face_crop, (200, 200))
#     return face_resized

# # --- STEP 2: Train LBPH model on first image and compare with second ---
# def compare_faces(img1_path, img2_path):
#     face1 = detect_face(img1_path)
#     face2 = detect_face(img2_path)

#     if face1 is None or face2 is None:
#         print("Face not detected properly.")
#         return

#     recognizer = cv2.face.LBPHFaceRecognizer_create()

#     # Train recognizer on first face
#     recognizer.train([face1], np.array([0]))

#     # Predict second face
#     label, confidence = recognizer.predict(face2)

#     print(f"Predicted label: {label}, Confidence: {confidence:.2f}")
#     if confidence < 60:  # Lower confidence → better match
#         print("✅ Same person (match)")
#     else:
#         print("❌ Different person")

# # --- STEP 3: Example usage ---
# compare_faces("pranay1.jpeg", "pranay2.jpeg")

import cv2
import os
import numpy as np

# --- STEP 0: Setup Haar Cascade ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- STEP 1: Detect and preprocess a face ---
def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"No face found in {image_path}")
        return None

    (x, y, w, h) = faces[0]  # Take first detected face
    face_crop = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, (200, 200))
    return face_resized

# --- STEP 2: Compare known images against test image ---
def compare_with_known_faces(known_folder, test_image_path):
    known_faces = []
    image_names = []

    # Load all known faces
    for file_name in os.listdir(known_folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(known_folder, file_name)
            face = detect_face(path)
            if face is not None:
                known_faces.append(face)
                image_names.append(file_name)

    if len(known_faces) == 0:
        print("❌ No valid known faces found.")
        return

    test_face = detect_face(test_image_path)
    if test_face is None:
        print("❌ No face found in test image.")
        return

    # Train recognizer on all known faces
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(known_faces, np.arange(len(known_faces)))

    label, confidence = recognizer.predict(test_face)

    print(f"🔍 Predicted label: {label}, Confidence: {confidence:.2f}")
    if confidence < 30:
        print(f"✅ Match found: {image_names[label]} (Confidence: {confidence:.2f})")
    else:
        print("❌ No matching face found in known set.")

# --- STEP 3: Run comparison ---
known_faces_folder = os.path.join(os.getcwd(), "arpit")   # Folder with known faces
test_image_path = "pranay2.jpeg"                              # Replace with your test image path

compare_with_known_faces(known_faces_folder, test_image_path)
