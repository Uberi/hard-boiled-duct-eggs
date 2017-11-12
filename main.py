#!/usr/bin/env python3

import face_recognition
import cv2
from numpy import array
import numpy as np
import time
from collections import deque, namedtuple
import math

class Tear:
    def __init__(self, xy, x_velocity, y_velocity, rot, scale, birth):
        self.xy = xy
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.rot = rot
        self.scale = scale
        self.birth = birth

tear_life = 20 # sec
tear_rate = 0.1 # 1 tear a sec
x_velocity_magnitude = 20 # sec
initial_y_velocity = -30
a = 10 # accel of g
T = deque()
last_step = time.time()
last_tear = -200
scale_rate = 1.8

WEBCAM_INDEX = 0
DEBUG = False

video_capture = cv2.VideoCapture(WEBCAM_INDEX)

def find_eye_outer_corners(left_eye_points, right_eye_points):
    face_axis = np.mean(right_eye_points, axis=0) - np.mean(left_eye_points, axis=0)
    left_index, right_index = np.argmin(left_eye_points @ face_axis), np.argmax(right_eye_points @ face_axis)
    return left_eye_points[left_index], right_eye_points[right_index]

def ndarray_to_coordinate(ndarray):
    return int(ndarray[0]), int(ndarray[1])

def step(tear, dt):
    tear.xy = (int(tear.x_velocity * dt + tear.xy[0]), int(tear.y_velocity * dt + tear.xy[1]))
    tear.y_velocity = a * dt + tear.y_velocity
    tear.rot = math.atan2(tear.y_velocity, tear.x_velocity)
    tear.scale = int(dt * scale_rate + tear.scale)

# time.time() is different for each tear, so might be better to either store 
# last update in each tear or just use 1 for all tears
def step_all(Q):
    global last_step
    global tear_life
    c = time.time()
    dt = c - last_step
    while Q and c - Q[0].birth > tear_life:
        Q.popleft()
    for t in Q:
        step(t, dt)
    last_step = c

def draw_tears(Q, frame):
    for t in Q:
        cv2.circle(frame,t.xy, t.scale, (255,191,0), -1)

while True:
    status, frame = video_capture.read()
    assert status

    face_landmarks_list = face_recognition.face_landmarks(frame)

    current = time.time()

    step_all(T)

    tear_generated = False

    for face_landmarks in face_landmarks_list:
        left_eye, right_eye = array(face_landmarks["left_eye"]), array(face_landmarks["right_eye"])
        left_corner, right_corner = find_eye_outer_corners(left_eye, right_eye)

        if current - last_tear > tear_rate:
            l_tear = Tear(ndarray_to_coordinate(left_corner), 
                          -x_velocity_magnitude,
                          initial_y_velocity,
                          math.atan2(initial_y_velocity, x_velocity_magnitude),
                          4,
                          current)
            r_tear = Tear(ndarray_to_coordinate(right_corner), 
                          x_velocity_magnitude,
                          initial_y_velocity,
                          math.atan2(initial_y_velocity, -x_velocity_magnitude),
                          4,
                          current)
            T.append(l_tear)
            T.append(r_tear)
            tear_generated = True

        if DEBUG:
            cv2.line(frame, ndarray_to_coordinate(left_corner), ndarray_to_coordinate(right_corner), (0, 0, 255), 4)

    if tear_generated:
        last_tear = current
        tear_generated = False

    draw_tears(T, frame)
    #wip: add drawing code here based on left_corner, right_corner

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
