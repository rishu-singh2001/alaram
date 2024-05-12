import cv2
import face_recognition
import playsound

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load alarm sound
alarm_sound = "test1.mp3"


# Function to play alarm sound
def play_alarm():
    playsound.playsound(alarm_sound)




# Load the known image (the face you want to match)
known_image = face_recognition.load_image_file("rs.jpg")

# Encode the known image
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for face_encoding in face_encodings:
        # Compare the face encoding of the current face with the known face encoding
        match = face_recognition.compare_faces([known_face_encoding], face_encoding)

        # If a match is found, print a message
        if match[0]:
            print("Match found!")
            break

            # You can add further actions here, such as displaying a message or saving the frame
        else:
            print("No match found.")
            play_alarm()

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
