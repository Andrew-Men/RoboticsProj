import time
import numpy as np
import cv2
import glob
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def read_node_real( reader, name ):
    node = reader.getNode( name )
    return node.real()

def read_node_string( reader, name ):
    node = reader.getNode( name )
    return node.string()

def read_node_matrix( reader, name ):
    node = reader.getNode( name )
    return node.mat()

# read image
#
raw_img = cv2.imread('IMG_6572.JPG')
plt.figure(1)
plt.imshow(raw_img)

# convert to gray scale image
gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

# read camera calibration parameters
camera_reader = cv2.FileStorage()
camera_reader.open("cameraParameters.xml",cv2.FileStorage_READ)
camera_matrix = read_node_matrix( camera_reader, "cameraMatrix" )
dist_coeffs = read_node_matrix( camera_reader, "dist_coeffs" )
camera_reader.release()

# define marker dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# define detector parameters (dont know the function of this variable)
parameters =  aruco.DetectorParameters_create()

# detect markers
corners, ids, _ = aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
if ids is not None:
    makers_img = aruco.drawDetectedMarkers(raw_img.copy(), corners, ids)
    plt.figure(2)
    plt.imshow(makers_img, origin='upper')

(rvecs, tvecs, _) = aruco.estimatePoseSingleMarkers(corners, 0.063, camera_matrix, dist_coeffs)

axis_img = raw_img.copy()
for i in range(ids.size):
    axis_img = aruco.drawAxis(axis_img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)

plt.figure(3)
plt.imshow(axis_img, origin='upper')
plt.show()
