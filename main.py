import numpy as np
import cv2
import matplotlib.pyplot as plt


haar_cascade_face = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


def face_detect(img):
    faces_rects = haar_cascade_face.detectMultiScale(
        img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img
