# from flask import Flask, request
# import datetime
# import os
# import cv2
# import numpy as np
# import face_recognition

# app = Flask(__name__)

# BASE_DIR = "recognised"
# os.makedirs(BASE_DIR, exist_ok=True)

# @app.route('/upload', methods=['POST'])
# def upload_image():

#     user_name = request.form.get('user', 'team_members')

#     user_dir = os.path.join(BASE_DIR, user_name)
#     os.makedirs(user_dir, exist_ok=True)

#     if 'image' not in request.files:
#         return "No image found in the request", 400

#     file = request.files['image']
#     img_bytes = np.frombuffer(file.read(), np.uint8)
#     img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

#     if img is None:
#         print("Failed to decode image.")
#         return "Invalid image file", 400

#     print(f"Image shape: {img.shape}")

#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     img_path = os.path.join(user_dir, f"{timestamp}.jpg")
#     cv2.imwrite(img_path, img)
#     print(f"Image saved at: {img_path}")

#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_img)
#     face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

#     if len(face_encodings) == 0:
#         print("No face detected in this image.")
#         return "No face detected", 400

#     encoding_path = os.path.join(user_dir, f"{timestamp}_encoding.npy")
#     np.save(encoding_path, face_encodings[0])
#     print(f"Encoding saved at: {encoding_path}")

#     return f"Image and encoding saved for user '{user_name}'", 200


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)



from flask import Flask, request
import datetime
import os
import cv2
import numpy as np
import insightface

app = Flask(__name__)

BASE_DIR = "recognised"
os.makedirs(BASE_DIR, exist_ok=True)

# Load ArcFace model once
print("Loading ArcFace model...")
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)  # CPU if no GPU

@app.route('/upload', methods=['POST'])
def upload_image():
    user_name = request.form.get('user', 'team_members')
    user_dir = os.path.join(BASE_DIR, user_name)
    os.makedirs(user_dir, exist_ok=True)

    if 'image' not in request.files:
        return "No image found in the request", 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        print("Failed to decode image.")
        return "Invalid image file", 400

    print(f"Image shape: {img.shape}")

    # Save original image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(user_dir, f"{timestamp}.jpg")
    cv2.imwrite(img_path, img)
    print(f"Image saved at: {img_path}")

    # Run ArcFace embedding
    faces = model.get(img)
    if len(faces) == 0:
        print("No face detected in this image.")
        return "No face detected", 400

    # Take the first face embedding
    embedding = faces[0].embedding
    encoding_path = os.path.join(user_dir, f"{timestamp}_encoding.npy")
    np.save(encoding_path, embedding)
    print(f"Encoding saved at: {encoding_path}")

    return f"Image and ArcFace encoding saved for user '{user_name}'", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
