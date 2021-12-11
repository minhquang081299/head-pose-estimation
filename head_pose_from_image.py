#!/usr/bin/env python3
import copy
import os
import time

import cv2
import sys
import dlib
# import argparse
import numpy as np

from drawFace import draw
import reference_world as world

PREDICTOR_PATH = "./models/shape_predictor_68_face_landmarks.dat"

# if not os.path.isfile(PREDICTOR_PATH):
#     print("[ERROR] USE models/downloader.sh to download the predictor")
#     sys.exit()

# parser = argparse.ArgumentParser()
#
# parser.add_argument("-i", "--image",
#                     type=str,
#                     help="image location for pose estimation")
#
# parser.add_argument("-f", "--focal",
#                     type=float,
#                     help="Callibrated Focal Length of the camera")
#
# args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def head_pose_estimation(im):
    # detector = dlib.get_frontal_face_detector()
    # time_s = 0
    # time_c1 = time.time()

    global predictor
    global detector
    # while True:
    # im = image.copy()
    time_i = time.time()
    faces = detector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0)
    print('time_i', time.time() - time_i)
    face3Dmodel = world.ref3DModel()

    for face in faces:
        # time_a = time.time()
        shape = predictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), face)
        # print('time_a', time.time() - time_a)
        draw(im, shape)

        refImgPts = world.ref2dImagePoints(shape)

        height, width, channel = im.shape
        focalLength = 1 * width
        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

        mdists = np.zeros((4, 1), dtype=np.float64)

        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(
            face3Dmodel, refImgPts, cameraMatrix, mdists)

        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        # noseEndPoint2D, jacobian = cv2.projectPoints(
        #     noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

        # draw nose line
        # p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        # p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        # cv2.line(im, p1, p2, (110, 220, 0),
        #          thickness=2, lineType=cv2.LINE_AA)

        # calculating angle
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        print('rl', angles[1])
        print('td', angles[2])
        print('tl', angles[0])
        # print('*' * 80)
        # print("Angle: ", angles)
        # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
        # x = np.arctan2(Qx[2][1], Qx[2][2])
        # y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
        # z = np.arctan2(Qz[0][0], Qz[1][0])
        # print("AxisX: ", x)
        # print("AxisY: ", y)
        # print("AxisZ: ", z)
        # print('*' * 10)
        # return angles[1]
        # gaze = "Looking: "
        # # threading = 0
        # # print('forward')
        # # if angles[0] > 50:
        # #     gaze += "Top"
        #
        # if angles[1] < -20:
        #     gaze += "Right"
        #     # threading = 0
        #     # print(gaze)
        #
        # elif angles[1] > 20:
        #     gaze += "Left"
        #     # threading = 1
        #     # print(gaze)
        # else:
        #     # gaze += "Forward"
        #     # print('Forward')
        #     if 100 < angles[0] < 175:
        #         gaze += "Top"
        #         # print(gaze)
        #     elif -100 > angles[0] > -175:
        #         gaze += "Bottom"
        #         # print(gaze)
        #     else:
        #         gaze += "Forward"
        # print(gaze)
        #
        threading = 0
        if angles[1] < -18:
            threading += 1
            # gaze += "Right"
            # threading = 0
            # print(gaze)

        elif angles[1] > 18:
            # gaze += "Left"
            # threading = 1
            # print(gaze)
            threading -= 1
        else:
            # gaze += "Forward"
            # print('Forward')
            if 100 < angles[0] < 173:
                threading += 2
                # gaze += "Top"
                # print(gaze)
            elif -100 > angles[0] > -173:
                threading -= 2
                # gaze += "Bottom"
                # print(gaze)
            else:
                threading += threading
                # gaze += "Forward"
        # cv2.putText(im, gaze, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        # cv2.imshow("Head Pose", im)
        return threading
    # head_pose_estimation(im)
    #     key = cv2.waitKey(10) & 0xFF
    #     if key == 27:
    #         cv2.imwrite(f"joye-{gaze}.jpg", im)
    #         break
    #
    # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main(args.image)
