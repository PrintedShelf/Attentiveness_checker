import cv2 
import dlib 
import imutils 
from imutils import face_utils
import os
import numpy as np


DEVICE_ID = 0 
capture = cv2.VideoCapture(DEVICE_ID)
predictor_path = os.getcwd() + "/data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(predictor_path) 

while(True): 
    ret, frame = capture.read() 
    frame = imutils.resize(frame, width=1000) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    rects = detector(gray, 0) 
    image_points = None
     
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape: 
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        image_points = np.array([
                tuple(shape[30]),#Nose tip
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
    
    if len(rects) > 0:
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

        size = frame.shape
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
        roll = eulerAngles[2]
        

        # cv2.putText(frame, 'yaw : ' + str(int(yaw)), (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        # cv2.putText(frame, 'pitch : ' + str(int(pitch)), (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        # cv2.putText(frame, 'roll : ' + str(int(roll)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)
        for p in image_points:
            cv2.drawMarker(frame, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)

        if yaw > 20 or yaw <= -20:
            cv2.putText(frame, "********Distracted**********", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        if pitch <= -30:
            cv2.putText(frame, "********Drowsy**********", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    cv2.imshow('frame',frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break


capture.release() 
cv2.destroyAllWindows() 