import http.server
import socketserver
import json
import os
import base64
import cv2
import numpy as np
import face_recognition

# --- Part 1: The Webpage (Your HTML with working JavaScript) ---
# This is your webpage code, now stored inside a Python string.
# The JavaScript has been completed to capture and send the image.

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Face Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f1f5f9; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .Register-container { background: white; padding: 30px 40px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 350px; text-align: center; }
        h2 { margin-bottom: 20px; font-size: 20px; font-weight: bold; }
        input[type="text"] { width: 100%; padding: 12px; border: 2px solid #cbd5e1; border-radius: 6px; margin-bottom: 20px; font-size: 14px; box-sizing: border-box;}
        #capture-button { display: none; } /* The original button is not needed for this flow */
        #canvas { display: none; } /* Canvas is for background capture, not for display */
        video { border-radius: 6px; margin-bottom: 20px; }
        button[type="submit"] { background-color: #2563eb; color: white; padding: 12px 20px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; width: 100%; }
        button[type="submit"]:hover { background-color: #1e40af; }
        #message { margin-top: 10px; font-size: 14px; color: #ef4444; font-weight: bold; }
    </style>
</head>
<body>
    <div class="Register-container">
        <h2>Register with Face Recognition</h2>
        <form id="register-form" method="post" enctype="multipart/form-data">
            <input type="text" name="username" id="username" placeholder="Username" required>
            <div class="camera-container">
                <video id="video" width="300" height="225" autoplay></video>
                <canvas id="canvas" width="300" height="225"></canvas>
            </div>
            <button type="submit">Register</button>
        </form>
        <div id="message"></div>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        const form = document.getElementById('register-form');
        const messageDiv = document.getElementById('message');

        // Start the webcam stream
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(err => {
                messageDiv.textContent = "Could not access webcam.";
                console.error("Webcam access error:", err);
            });

        // This function runs when the "Register" button is clicked
        form.addEventListener('submit', function(event) {
            event.preventDefault(); // Stop the page from reloading
            
            const username = document.getElementById('username').value;
            if (!username.trim()) {
                messageDiv.textContent = 'Please enter a username.';
                return;
            }

            // 1. Capture a still frame from the video
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageDataURL = canvas.toDataURL('image/jpeg');

            // 2. Send the captured image and username to the Python server
            fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: username, image: imageDataURL })
            })
            .then(response => response.json())
            .then(data => {
                // 3. Display the response message from the Python server
                messageDiv.textContent = data.message;
                messageDiv.style.color = data.status === 'success' ? '#22c55e' : '#ef4444';
            })
            .catch(error => {
                messageDiv.textContent = 'Failed to connect to the server.';
                console.error('Server communication error:', error);
            });
        });
    </script>
</body>
</html>
"""

# --- Part 2: The Backend Logic ---
# This section contains the adapted version of your Python script.

KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

def register_face(username, data_url):
    """
    Processes a registration request.
    This function contains the core face encoding logic from your original script.
    """
    # Check if a user with this name already exists
    if os.path.exists(os.path.join(KNOWN_FACES_DIR, f"{username}.npy")):
        return ("error", "A user with this name already exists.")
        
    try:
        # Convert the image sent from the browser (data URL) into an OpenCV image
        header, encoded = data_url.split(",", 1)
        decoded_data = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Use the face_recognition library to find the face encoding
        # This is the same core logic from your original script
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if len(face_encodings) == 1:
            # If exactly one face is found, save its encoding
            face_encoding = face_encodings[0]
            encoding_path = os.path.join(KNOWN_FACES_DIR, f"{username}.npy")
            np.save(encoding_path, face_encoding)
            print(f"Successfully registered {username}.")
            return ("success", f"User '{username}' registered successfully!")
        elif len(face_encodings) > 1:
            print("Registration failed: more than one face detected.")
            return ("error", "Multiple faces detected. Please show only one face.")
        else:
            print("Registration failed: no face detected.")
            return ("error", "No face could be detected in the image.")

    except Exception as e:
        print(f"An error occurred during registration: {e}")
        return ("error", "An internal server error occurred.")

# --- Part 3: The Simple Web Server ---
# This minimal server bridges the gap between the webpage and the Python code.

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # When the user opens the page, serve the HTML code
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        else:
            super().do_GET()

    def do_POST(self):
        # When the webpage sends data (the image), this function runs
        if self.path == '/register':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            # Call our registration function with the received data
            status, message = register_face(data['username'], data['image'])

            # Send the result back to the webpage's JavaScript
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': status, 'message': message}).encode())
        else:
            self.send_error(404)

# --- Main execution block ---
PORT = 8000
print(f"Server starting at http://localhost:{PORT}")
print("Go to this address in your web browser to register a face.")
print(f"Registered faces will be saved in the '{KNOWN_FACES_DIR}' folder.")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()