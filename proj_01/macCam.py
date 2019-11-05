import numpy as np
import cv2
import glob
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time

def read_node_real( reader, name ):
    node = reader.getNode( name )
    return node.real()

def read_node_string( reader, name ):
    node = reader.getNode( name )
    return node.string()

def read_node_matrix( reader, name ):
    node = reader.getNode( name )
    return node.mat()

# start image capture
cap = cv2.VideoCapture(0)

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

# read image
#
while(True):
    time.sleep(0.1)

    # get image
    ret, raw_img = cap.read()

    # convert to gray scale image
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    # detect markers
    corners, ids, _ = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
    if ids is not None:
        (rvecs, tvecs, _) = aruco.estimatePoseSingleMarkers(corners, 0.063, camera_matrix, dist_coeffs)
        axis_img = raw_img.copy()
        for i in range(ids.size):
            axis_img = aruco.drawAxis(axis_img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)
            if cv2.waitKey(100) == 115 : # check for s key
                print(ids[i])
                print(rvecs[i])
                print(tvecs[i])
                print()

        r = np.zeros((1,3),dtype='float64')
        t = np.zeros((1,3),dtype='float64')
        axis_img = aruco.drawAxis(axis_img, camera_matrix, dist_coeffs, r, t, 0.05)
        cv2.imshow('Image with Axis', axis_img)

        if cv2.waitKey(100) == 113 : # check for q key
            break

cap.release()
cv2.destroyAllWindows()
