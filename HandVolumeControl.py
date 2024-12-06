import cv2
import time
import math
import numpy as np
import Tracker as trk


# Set Camera size
cam_width, cam_height = 640, 480

# store the Hand tracking module
tracker = trk.HandDetector(min_detection_confi=0.7)


# init camera
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
prev_time = 0

while True:
    success, img = cap.read()
    # get hand image
    img = tracker.findHands(img)
    lm_list = tracker.findPosition(img, draw=False)
    # get landmark of points to be used for tracking
    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw a circle around the points and a line between them

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        # Get the length between the points
        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        if length < 18:
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

    # Set FPS
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    # add FPS value to image
    cv2.putText(
        img, f"FPS: {int(fps)}", (40, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3
    )

    # Display Camera
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
