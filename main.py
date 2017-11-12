#!/usr/bin/env python3

import face_recognition
import cv2
from numpy import array
import numpy as np

from PIL import Image, ImageDraw

WEBCAM_INDEX = 0
DEBUG = True

video_capture = cv2.VideoCapture(WEBCAM_INDEX)

def find_eye_outer_corners(left_eye_points, right_eye_points):
    face_axis = np.mean(right_eye_points, axis=0) - np.mean(left_eye_points, axis=0)
    left_index, right_index = np.argmin(left_eye_points @ face_axis), np.argmax(right_eye_points @ face_axis)
    return left_eye_points[left_index], right_eye_points[right_index]

def ndarray_to_coordinate(ndarray):
    return int(ndarray[0]), int(ndarray[1])

while True:
    status, frame = video_capture.read()
    assert status

    face_landmarks_list = face_recognition.face_landmarks(frame)

    for face_landmarks in face_landmarks_list:
        left_eye, right_eye = array(face_landmarks["left_eye"]), array(face_landmarks["right_eye"])
        left_corner, right_corner = find_eye_outer_corners(left_eye, right_eye)

        if DEBUG:
            cv2.line(frame, ndarray_to_coordinate(left_corner), ndarray_to_coordinate(right_corner), (0, 0, 255), 4)

        #wip: add drawing code here based on left_corner, right_corner


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()