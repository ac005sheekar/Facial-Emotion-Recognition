########       SHEEKAR BANERJEE      #########
########       AI LEAD ENGINEER      #########

#using python 3.8.3


import cv2
from deepface import DeepFace  #py -m pip uninstall deepface         #py -m pip install deepface==0.0.75

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Webcam cannot be opened...")

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions = ['emotion'])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow('Original Video', frame)



    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

    print(result[0]["dominant_race"][:])
    print(result[0]["dominant_emotion"][:])
    print(result[0]["age"])
    print(result[0]["dominant_gender"][:])

cap.release()
cv2.destroyAllWindows()