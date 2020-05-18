"""
Disparity map calculation from the vehicle stereo camera pair.

"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


SWS = 5
PFS = 5
PFC = 29
MDS = -25
NOD = 128
TTH = 100
UR = 10
SR = 15
SPWS = 100


stereo = cv2.StereoSGBM_create(0,numDisparities=16, blockSize=21)

'''
stereo.setPreFilterType(1)
stereo.setPreFilterSize(PFS)
stereo.setPreFilterCap(PFC)
stereo.setMinDisparity(MDS)
stereo.setNumDisparities(NOD)
stereo.setTextureThreshold(TTH)
stereo.setUniquenessRatio(UR)
stereo.setSpeckleRange(SR)
stereo.setSpeckleWindowSize(SPWS)
'''

def calculate_disparity_map(img_left, img_right):
    disparity1 = stereo.compute(img_left, img_right)
    disparity2 = stereo.compute(img_right, img_left)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #disparity = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
    local_max1 = disparity1.max()
    local_min1 = disparity1.min()
    local_max2 = disparity2.max()
    local_min2 = disparity2.min()
    cv2.imshow("disparity1", disparity1)  
    cv2.imshow("disparity2", disparity2)  
    disparity_visual1 = (disparity1-local_min1)*(1.0/(local_max1-local_min1))
    disparity_visual2 = (disparity2-local_min2)*(1.0/(local_max2-local_min2))
    
    #cv8uc = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow("disparityv1", disparity_visual1)  
    cv2.imshow("disparityv2", disparity_visual2)  
    return disparity1


def plot_disparity_map(img_left, img_right, disparity):

    cv2.imshow('Left camera view', img_left)
    cv2.imshow('Right camera view', img_right)
    cv2.imshow('Calculated disparity map', disparity)

# This part is run when the script is executed, but not imported
if __name__ == '__main__':

    for i in range(0, 8):
        imgL = cv2.imread('sawtooth/im' + str(i) + '.ppm', 0)
        imgR = cv2.imread('sawtooth/im' + str(i + 1) + '.ppm', 0)
        '''
        imgL = cv2.imread('ohta/scene1.row' + str(j) + '.col' + str(i) + '.ppm', 0)
        imgR = cv2.imread('ohta/scene1.row' + str(j) + '.col' + str(i + 1) + '.ppm', 0)
        '''
        disparity = calculate_disparity_map(imgL, imgR)
        #plot_disparity_map(imgL, imgR, disparity)

        cv2.waitKey(2000)
