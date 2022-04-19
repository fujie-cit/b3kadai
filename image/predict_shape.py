import dlib
from imutils import face_utils
import cv2
import numpy as np

def proc(imgname): 
    face_detector = dlib.get_frontal_face_detector()
    
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    img = cv2.imread(imgname)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector(img_gry, 1)

    for face in faces:
        landmark = face_predictor(img_gry, face)
        landmark = face_utils.shape_to_np(landmark)

        for (i, (x, y)) in enumerate(landmark):
            cv2.circle(img, (x, y), 5, (255, 0 , 0), -1)
        break

    return img, landmark

if __name__ == '__main__':
    filename = 'S022_006_00000001.png'
    img, landmarks = proc(filename)
    cv2.imwrite('001.png', img)
    np.save('001.npy', landmarks)
    print(landmarks)

    filename = 'S022_006_00000017.png'
    img, landmarks = proc(filename)
    cv2.imwrite('017.png', img)
    np.save('017.npy', landmarks)
    print(landmarks)
