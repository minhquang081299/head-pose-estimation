import copy

import cv2
import imutils
import time
from head_pose_from_image import head_pose_estimation
from imutils.video import VideoStream

# cap = cv2.VideoCapture(0)
cap = VideoStream(src=0).start()
# cap = cv2.VideoCapture('rtsp://admin:Hungha123@123.24.142.145/H265?ch=1&subtype=0')
while True:
    frame = cap.read()
    image = cv2.flip(frame, 1)
    # im = image.copy()
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # h, w, c = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # image.flags.writeable = False
    # results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # time_process = 0
    time_c = time.time()
    # if not ret:
    #     print('Error camera!!')
    #
    # else:
    #     continue
    # head_pose_estimation(image)

    threading = head_pose_estimation(image)
    gaze = 'Look: '
    if threading == 1:
        gaze += 'right'
    elif threading == -1:
        gaze += 'left'
    else:
        if threading == 2:
            gaze += 'top'
        elif threading == -2:
            gaze += 'bottom'
        else:
            gaze += 'forward'

    print('time_process:', time.time() - time_c)
    print(gaze)
    # cv2.imshow('Head-pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
