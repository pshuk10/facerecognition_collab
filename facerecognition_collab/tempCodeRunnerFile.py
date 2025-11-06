from flask import Flask, request
import datetime
import os
import cv2
import numpy as np

app = Flask(__name__)
SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image found", 400
    
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        print("❌ Failed to decode image.")
        return "Invalid image", 400

    # Print shape (height, width, channels)
    print(f"📏 Image shape: {img.shape}")

    # Save the image
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, img)
    print(f"✅ Image saved as {path}")

    return "Image received successfully", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
