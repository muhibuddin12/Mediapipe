from math import radians
import cv2
from cv2 import circle
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## Video Feed
cap = cv2.VideoCapture(0)

# function calculate angles
def calculate_angle(a,b,c):
    a = np.array(a) #Titik 1
    b = np.array(b) #titik 2
    c = np.array(c) #titik 3

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #make detection
        results = pose.process(image)

        #recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extract Landmark
        try:
            landmarks = results.pose_landmarks.landmark
            
            #get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            #calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            print(angle)

            #visualize
            cv2.putText(image, str(angle), tuple(np.multiply(elbow,[640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2,cv2.LINE_AA)
        except:
            pass


        #render detection
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(234,115,60), thickness=1, circle_radius=3),
            mp_drawing.DrawingSpec(color=(112,115,60), thickness=2, circle_radius=3)
        )

        

        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break;


# for lndmrk in mp_pose.PoseLandmark:
#     print(lndmrk)
# print position by cam
# shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
# print(np.multiply(shoulder,[640,480]))

cap.realease()
cv2.destroyAllWindows()
