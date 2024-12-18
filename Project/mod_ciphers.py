import cv2
import time

# Load the face detection model
face_cap = cv2.CascadeClassifier("C:/Users/dell/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)

# List to keep track of attendance
attendance = []
face_id_counter = 0

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Failed to capture video. Exiting..")
        break

    # Flip the video horizontally (for mirror effect)
    video_data = cv2.flip(video_data, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Limit to 3 faces
    if len(faces) > 3:
        faces = faces[:3]

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Mark attendance for each face (use a unique face ID or timestamp)
        face_id = f"Face_{face_id_counter}"
        if face_id not in attendance:
            attendance.append(face_id)
            face_id_counter += 1
            print(f"Marked attendance for {face_id}")
    
    # Display the video frame with rectangles
    cv2.imshow("Video_Live", video_data)

    # Display attendance on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(video_data, f"Attendance: {len(attendance)}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Exit the loop when 'a' is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Print the final attendance list when the program ends
print("Final Attendance List:")
for person in attendance:
    print(person)

# Release the video capture and close all windows
video_cap.release()
cv2.destroyAllWindows()
