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
plt.ion()
raw_img = cv2.imread('IMG_9030.jpg')
plt.figure(1)
plt.imshow(raw_img)
plt.show()

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
    plt.show()
    plt.pause(1000)