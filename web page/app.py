from flask import Flask, render_template, request
import cv2
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_fingerprint_input(file_storage):
    image_path = os.path.join(UPLOAD_FOLDER, file_storage.filename)
    file_storage.save(image_path)
    image = cv2.imread(image_path, 0)
    if image is None:
        print("Could not open or find the image.")
        return None
    return image

def compare_fingerprints(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches) > 200

stored_images = {}

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    if request.method == "POST":
        username = request.form.get("username")
        action = request.form.get("action")
        file = request.files["fingerprint"]
        if not file:
            message = "No fingerprint file uploaded."
        else:
            image = get_fingerprint_input(file)
            if image is None:
                message = "Could not read fingerprint image."
            elif action == "register":
                stored_images[username] = image
                message = f"User '{username}' registered successfully."
            elif action == "login":
                stored_image = stored_images.get(username)
                if stored_image is not None and compare_fingerprints(image, stored_image):
                    message = f"Login successful for user '{username}'."
                else:
                    message = f"Login failed for user '{username}'."

    return render_template("index.html", message=message)

if __name__ == "__main__":
    app.run(debug=True)