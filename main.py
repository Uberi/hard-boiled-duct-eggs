#!/usr/bin/env python3

import face_recognition
import cv2

from PIL import Image, ImageDraw

WEBCAM_INDEX = 0

video_capture = cv2.VideoCapture(WEBCAM_INDEX)

while True:
    status, frame = video_capture.read()
    assert status

    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        d.line(face_landmarks[facial_feature], width=5)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()