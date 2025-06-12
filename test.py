import face_recognition
import cv2
import numpy as np
import sys

KNOWN_PERSON_IMAGE_PATH = r"C:\Users\surbh\Desktop\FaceLoginSys\FaceLogin\test_image.jpg"
# The name you want to assign to the recognized person
KNOWN_PERSON_NAME = "Matched!" 

# 1. Load the known image and create an embedding
try:
    known_image = face_recognition.load_image_file(KNOWN_PERSON_IMAGE_PATH)
    # The face_encodings function returns a list of encodings. 
    # Since we know our image has one face, we take the first one [0].
    known_face_encodings = face_recognition.face_encodings(known_image)

    if not known_face_encodings:
        print(f"Error: No face found in the image at '{KNOWN_PERSON_IMAGE_PATH}'.")
        print("Please use a clear picture of a face.")
        sys.exit()

    known_face_encoding = known_face_encodings[0]

except FileNotFoundError:
    print(f"Error: Could not find the image file '{KNOWN_PERSON_IMAGE_PATH}'.")
    print("Please make sure the image is in the same folder as the script.")
    sys.exit()

# 2. Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

print("Webcam started. Looking for faces...")
print("Press 'q' to quit.")

# --- MAIN LOOP ---

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:
        break

    # Find all the faces and face encodings in the current frame of video
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    live_face_locations = face_recognition.face_locations(rgb_frame)
    live_face_encodings = face_recognition.face_encodings(rgb_frame, live_face_locations)

    # Loop through each face found in the live frame
    for (top, right, bottom, left), live_face_encoding in zip(live_face_locations, live_face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([known_face_encoding], live_face_encoding)

        name = "Unknown"
        color = (0, 0, 255) # Red for unknown

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            name = KNOWN_PERSON_NAME
            color = (0, 255, 0) # Green for known

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Real-time Face Recognition', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()