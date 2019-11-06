import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import time

# function for read camera_matrix and dist_coeffs
def read_node_matrix(reader, name):
    node = reader.getNode(name)
    return node.mat()

# read camera calibration parameters
camera_reader = cv2.FileStorage()
camera_reader.open("macCameraParameters.xml",cv2.FileStorage_READ)
camera_matrix = read_node_matrix( camera_reader, "cameraMatrix" )
dist_coeffs = read_node_matrix( camera_reader, "dist_coeffs" )
camera_reader.release()

# define marker dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# define detector parameters (dont know the function of this variable)
parameters =  aruco.DetectorParameters_create()

flag = 0

# start image capture
cap = cv2.VideoCapture(0)
while(True):
    time.sleep(0.1) # reduce capture rate to save compute resource

    # get image from computer camera
    ret, raw_img = cap.read()
    axis_img = raw_img.copy()

    # convert to gray scale image
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    # detect markers
    corners, ids, _ = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)

    # detect for key 's'
    if cv2.waitKey(10) == 115 :
        flag = 1
    if ids is not None:
        # calculate marker pose
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.063, camera_matrix, dist_coeffs)

        # draw marker frame axis onto the image
        for i in range(ids.size):
            axis_img = aruco.drawAxis(axis_img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)

            # if key 's' is pressed, print out rvecs and tvecs
            if flag == 1 :
                print(ids[i])
                print(rvecs[i])
                print(tvecs[i])
                print()
                flag = 0

    # draw the spatial frame(Frame S) on the screen for calibrate the spatical frame
    r = np.zeros((1,3),dtype='float32')     # the spatical frame is translated from camera frame 
    t = np.asarray([0.0,0.0,1.0])                 # and the v = [0,0,1] (Unit: meter); omega = [0,0,0]
    axis_img = aruco.drawAxis(axis_img, camera_matrix, dist_coeffs, r, t, 0.05)

    # display image
    cv2.imshow('Image with Axis', axis_img)

    if cv2.waitKey(10) == 113 : # check for q key
        break

cap.release()
cv2.destroyAllWindows()
