import face_recognition
import cv2
import sys

KNOWN_PERSON_IMAGE_PATH = r"C:\Users\surbh\Desktop\FaceLoginSys\FaceLogin\test_image.jpg"

try:
    known_image = face_recognition.load_image_file(KNOWN_PERSON_IMAGE_PATH)
  
    known_face_encodings = face_recognition.face_encodings(known_image)

    if not known_face_encodings:
        print(f"Error: No face found in the image at '{KNOWN_PERSON_IMAGE_PATH}'.")
        
        sys.exit()

    known_face_encoding = known_face_encodings[0]

except FileNotFoundError:
    print(f"Error: Could not find the image file '{KNOWN_PERSON_IMAGE_PATH}'.")

    sys.exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

print("Press 'q' to quit.")



while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    live_face_locations = face_recognition.face_locations(rgb_frame)
    live_face_encodings = face_recognition.face_encodings(rgb_frame, live_face_locations)

    for (top, right, bottom, left), live_face_encoding in zip(live_face_locations, live_face_encodings):
        matches = face_recognition.compare_faces([known_face_encoding], live_face_encoding)

        name = "Unknown"
        color = (0, 0, 255)


        if True in matches:
            name = "Matched!"
            color = (0, 255, 0) 

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    
    cv2.imshow('Real-time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()