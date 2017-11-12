#!/usr/bin/env python3

import face_recognition
import cv2
from numpy import array
import numpy as np
import time
from collections import deque, namedtuple
import math

Tear = namedtuple("Tear", ["x", "y", "x_velocity", "y_velocity", "rot", "scale", "birth"])

tear_life = 3 # sec
x_velocity_magnitude = 5 # sec
a = 9.8 # accel of g
T = deque()
last_t = None

WEBCAM_INDEX = 0
DEBUG = True

video_capture = cv2.VideoCapture(WEBCAM_INDEX)

def find_eye_outer_corners(left_eye_points, right_eye_points):
    face_axis = np.mean(right_eye_points, axis=0) - np.mean(left_eye_points, axis=0)
    left_index, right_index = np.argmin(left_eye_points @ face_axis), np.argmax(right_eye_points @ face_axis)
    return left_eye_points[left_index], right_eye_points[right_index]

def ndarray_to_coordinate(ndarray):
    return int(ndarray[0]), int(ndarray[1])

def step(tear, dt):
    tear.x = tear.x_velocity * dt + tear.x
    tear.y = tear.y_velocity * dt + tear.y
    tear.y_velocity = a * dt * tear.y_velocity
    tear.rot = math.atan2(tear.y_velocity, tear.x_velocity)
    # TODO scale

# time.time is different for each tear, so might be better to either store 
# last update in each tear or just use 1 for all tears
def step_all(Q):
    global last_t
    global tear_life
    c = time.time
    dt = last_t - c
    while c - Q[0].birth > tear_life:
        Q.popleft()
    for t in Q:
        step(t, dt)
    last_t = c

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
