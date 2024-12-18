import cv2
import time


face_cap = cv2.CascadeClassifier("C:/Users/dell/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)


attendance = []
tracked_faces = [] 


max_faces = 3

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Failed to capture video. Exiting..")
        break

   
    video_data = cv2.flip(video_data, 1)

   
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    if len(faces) > max_faces:
        faces = faces[:max_faces]

    
    for (x, y, w, h) in faces:
       
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        face_box = (x, y, w, h)
        if face_box not in tracked_faces:
            
            tracked_faces.append(face_box)
            attendance.append(f"Face_{len(tracked_faces)}")
            print(f"Marked attendance for Face_{len(tracked_faces)}")

   
    cv2.imshow("Video_Live", video_data)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(video_data, f"Attendance: {len(attendance)}", (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    
    if cv2.waitKey(10) == ord("a"):
        break


print("Final Attendance List:")
for person in attendance:
    print(person)

video_cap.release()
cv2.destroyAllWindows()