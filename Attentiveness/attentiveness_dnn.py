import os
import cv2
import imutils
import numpy as np
import tensorflow as tf
from imutils import face_utils
from scipy.spatial import distance
from scipy.spatial import ConvexHull
from tensorflow import keras

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

def get_face_detector(modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel",
                      configFile = "models/deploy.prototxt"):
    modelFile = os.getcwd()+"/models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = os.getcwd()+"/models/deploy.prototxt"
    model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

def get_landmark_model(saved_model=os.getcwd()+'/models/pose_model'):
    model = tf.saved_model.load(saved_model)
    return model

def get_square_box(box):
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   
        return box
    elif diff > 0:                  
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

def move_box(box, offset):
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

def detect_marks(img, model, face):
    offset_y = int(abs((face[3] - face[1]) * 0.1))
    box_moved = move_box(face, [0, offset_y])
    facebox = get_square_box(box_moved)
    
    face_img = img[facebox[1]: facebox[3],
                     facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    predictions = model.signatures["predict"](
        tf.constant([face_img], dtype=tf.uint8))

    marks = np.array(predictions['output']).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))
    
    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    marks = marks.astype(np.uint)

    return marks

def draw_marks(image, marks, color=(255, 0, 0)):
    count = 0
    for mark in marks:
        count+=1
#         print(f'{count} : {mark[0]},{mark[1]}')
        cv2.circle(image, (mark[0], mark[1]), 2, color, -1, cv2.LINE_AA)
#        cv2.putText(image, f'{count}', (mark[0], mark[1]), cv2.FONT_HERSHEY_SIMPLEX, 
#                    0.5, (255, 0, 0), 0, cv2.LINE_AA)


        
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]        
face_model = get_face_detector()
landmark_model = get_landmark_model()
flag_for_ear=0
position_drowsy_frame = 0
position_distracted_frame = 0
position_frame_threshold = 10
frame_check = 10
cap=cv2.VideoCapture(0)
ear_threshold = 0.22
while True:
    ret,img = cap.read()
    img = imutils.resize(img, width=450)
    rects = find_faces(img, face_model)
    for rect in rects:
        marks = detect_marks(img, landmark_model, rect)
#         print(marks)
        image_points = np.array([
                    tuple(marks[30]),
                    tuple(marks[21]),
                    tuple(marks[22]),
                    tuple(marks[39]),
                    tuple(marks[42]),
                    tuple(marks[31]),
                    tuple(marks[35]),
                    tuple(marks[48]),
                    tuple(marks[54]),
                    tuple(marks[57]),
                    tuple(marks[8]),
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
    #             for p in image_points:
    #                 cv2.drawMarker(img, (int(p[0]), int(p[1])),  (0.0, 1.409845, 255),markerType=cv2.MARKER_CROSS, thickness=1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2)

            if yaw > 20 or yaw <= -20:
                position_distracted_frame += 1
                if position_distracted_frame > position_frame_threshold:
                    cv2.putText(img, "********Distracted**********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                 position_distracted_frame = 0   
            if pitch <= -30:
                position_drowsy_frame += 1
                if position_drowsy_frame > position_frame_threshold:
                    cv2.putText(img, "********Drowsy**********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                position_drowsy_frame = 0   
    
    
        draw_marks(img, marks)
        leftEye = marks[lStart:lEnd]
        rightEye = marks[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye.tolist())
        rightEAR = eye_aspect_ratio(rightEye.tolist())
        lctr = np.array(leftEye.tolist()).reshape((-1,1,2)).astype(np.int32)
        rctr = np.array(rightEye.tolist()).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [lctr], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rctr], -1, (0, 255, 0), 1)
        ear = (leftEAR + rightEAR) / 2.0
#         print(leftEAR,rightEAR)
#         print(ear)

        if ear < ear_threshold:
            flag_for_ear += 1
            if flag_for_ear >= frame_check:
                cv2.putText(img, "********Drowsy**********", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag_for_ear = 0

        lips = [60,61,62,63,64,65,66,67]
        frame_counter = 0
        lip_lar = 0.35
        lip_point = marks[lips]
        lar = calculate_lip(lip_point.tolist()) 
#         cv2.putText(img, f'lar : {lar}', (10, 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 2)
        if lar > lip_lar:
            cv2.putText(img, "********Drowsy**********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)            
    cv2.imshow("image", img)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()
