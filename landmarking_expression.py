import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/khushboogupta/Downloads/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

name = input("Enter your expression: ")

frames  = []
output = []

while True:
    ret,frame  = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret:

        faces  = detector(gray)

        for face in faces:
            landmarks = predictor(gray,face)
            nose = landmarks.parts()[28]
            # print(nose.x, nose.y)

            # lip_up = landmarks.parts()[63].y
            # lip_down = landmarks.parts()[67].y
            #
            # if lip_down-lip_up>5:
            #     print ("mouth open")
            # else:
            #     print("mouth closed")

            expressions = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])
            # print(expressions.flatten())
            #
            # for point in landmarks.parts()[48:]:
            #     cv2.circle(frame, (point.x, point.y), 1 ,(255,0,0), 2)


        cv2.imshow("My screen",frame)


    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("c"):

        frames.append(expressions.flatten())
        output.append([name])

x = np.array(frames)
y = np.array(output)

data = np.hstack([y, x])

f_name = "face_expression.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)

cap.release()
cv2.destroyAllWindows()
