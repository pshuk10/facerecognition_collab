from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/upload', methods=['POST'])
def upload_image():
    device_id = request.form.get('device_id', 'unknown_device')
    mode = request.form.get('mode', 'undefined_mode')
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({'status': 'error', 'message': 'No image uploaded'}), 400

    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({
            'status': 'success',
            'device_id': device_id,
            'mode': mode,
            'faces_detected': 0,
            'message': 'No face detected'
        })

    recognition_results = []
    for (x, y, w, h) in faces:
        face_crop = img[y:y+h, x:x+w]
        try:
            result = DeepFace.find(
                img_path=face_crop,
                db_path="known_faces/",
                model_name="Facenet",
                enforce_detection=False
            )
            if len(result) > 0:
                match = result.iloc[0]["identity"].split("/")[-1]
                recognition_results.append(match)
            else:
                recognition_results.append("Unknown")
        except Exception as e:
            recognition_results.append("Error: " + str(e))

    return jsonify({
        'status': 'success',
        'device_id': device_id,
        'mode': mode,
        'faces_detected': len(faces),
        'recognized_faces': recognition_results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
