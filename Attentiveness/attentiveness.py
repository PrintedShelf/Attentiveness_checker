from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import os
import cv2
import numpy as np
import time


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def calculate_lip(lips):
     dist1 = distance.euclidean(lips[2], lips[6]) 
     dist2 = distance.euclidean(lips[0], lips[4]) 
     lar = float(dist1/dist2)
     return lar

def detect_drowsiness(thresh):
     ear_threshold = thresh - 0.07
     frame_check = 15
     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

     cap=cv2.VideoCapture(0)
     flag_for_ear=0

     while True:
        ret, img=cap.read()
        img = imutils.resize(img, width=450)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        txt='Not Attentive'
        for subject in subjects:
            #Head-pose estimation
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape: 
                cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
            image_points = np.array([
                    tuple(shape[30]),
                    tuple(shape[21]),
                    tuple(shape[22]),
                    tuple(shape[39]),
                    tuple(shape[42]),
                    tuple(shape[31]),
                    tuple(shape[35]),
                    tuple(shape[48]),
                    tuple(shape[54]),
                    tuple(shape[57]),
                    tuple(shape[8]),
                    ],dtype='double')
            if len(subjects) > 0:
                model_points = np.array([
                (0.0,0.0,0.0), 
                (-30.0,-125.0,-30.0), 
                (30.0,-125.0,-30.0), 
                (-60.0,-70.0,-60.0), 
                (60.0,-70.0,-60.0),
                (-40.0,40.0,-50.0), 
                (40.0,40.0,-50.0), 
                (-70.0,130.0,-100.0), 
                (70.0,130.0,-100.0), 
                (0.0,158.0,-10.0), 
                (0.0,250.0,-50.0) 
                ])
                size = img.shape
                focal_length = size[1]
                center = (size[1] // 2, size[0] // 2) 

                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype='double')

                dist_coeffs = np.zeros((4, 1))
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
                mat = np.hstack((rotation_matrix, translation_vector))
                (_, _, _, _, _, _, eulerAngles) = cv2.decomposeProjectionMatrix(mat)
                yaw = eulerAngles[1]
                pitch = eulerAngles[0]
                # roll = eulerAngles[2]
                (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                            translation_vector, camera_matrix, dist_coeffs)
                for p in image_points:
                    cv2.drawMarker(img, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2)

                if yaw > 20 or yaw <= -20:
                    cv2.putText(img, "********Distracted**********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if pitch <= -30:
                    cv2.putText(img, "********Drowsy**********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Calculating EAR and detecting drowsiness

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < ear_threshold:
                flag_for_ear += 1
                if flag_for_ear >= frame_check:
                    cv2.putText(img, "********Drowsy**********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag_for_ear = 0

            # Calculating MAR and detecting drowsiness
            lips = [60,61,62,63,64,65,66,67]
            frame_counter = 0
            lip_lar = 0.35
            lip_point = shape[lips]
            lar = calculate_lip(lip_point) 
            if lar > lip_lar:
                frame_counter += 1
                if frame_counter > lip_lar:
                    cv2.putText(img, "********Drowsy**********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_counter = 0
                
        cv2.imshow("Frame",img)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
     cv2.destroyAllWindows()
     cap.release() 
     
def calibrate():
     thresh = []
     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

     cap=cv2.VideoCapture(0)
     capture_duration = 5
     start_time = time.time()
     while( int(time.time() - start_time) < capture_duration ):
        ret, img=cap.read()
        img = imutils.resize(img, width=450)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            structure = predict(gray, subject)
            structure = face_utils.shape_to_np(structure)
            leftEye = structure[lStart:lEnd]
            rightEye = structure[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
            thresh.append(ear)
            cv2.putText(img, "Calibrating! (Try not to blink)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame",img)
        key = cv2.waitKey(10) & 0xFF  
     cv2.destroyAllWindows()
     cap.release()
     if len(thresh):
         return sum(thresh)/len(thresh)
     else:
         return 0
     

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(os.getcwd() +"/data/shape_predictor_68_face_landmarks.dat")
get_thresh = calibrate()
if get_thresh:
    detect_drowsiness(get_thresh)
else:
    print('Try recalibrating!!!')