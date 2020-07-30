import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/khushboogupta/Downloads/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret,frame  = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret:

        faces  = detector(gray)

        for face in faces:
            landmarks = predictor(gray,face)
            nose = landmarks.parts()[28]
            # print(nose.x, nose.y)
            for point in landmarks.parts():
                cv2.circle(frame, (point.x, point.y), 1 ,(255,0,0),2)

        print (faces)

        cv2.imshow("My screen",frame)


    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
