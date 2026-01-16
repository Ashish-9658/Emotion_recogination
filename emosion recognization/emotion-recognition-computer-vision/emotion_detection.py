import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smiles = smile_cascade.detectMultiScale(gray, 2.6, 30)
    eyes = eye_cascade.detectMultiScale(gray, 1.5, 7)

   
    if len(faces) == 0:
        cv2.putText(frame, "Face Not Detected", (360, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.putText(frame, "Face Detected", (400, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    
    if len(smiles) == 0:
        cv2.putText(frame, "Normal", (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 1)
            cv2.putText(frame, "Smiling", (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    
    if len(eyes) == 0:
        cv2.putText(frame, "Eyes Closed", (400, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    else:
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), 1)
        cv2.putText(frame, "Eyes Opened", (400, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return frame


video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
