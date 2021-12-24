# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:34:51 2021

@author: VIVEK RUGLE
"""
#Object Tracking with bounding boxes

import cv2
import numpy as np

load_from_disk = True
 
if load_from_disk:
    penval = np.load('penval.npy')

#penval = [[100, 150, 80], [140, 255, 255]]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
 
# kernel for morphological operations
kernel = np.ones((5,5),np.uint8)
 
# set the window to auto-size so we can view this full screen.
cv2.namedWindow('image', cv2.WINDOW_NORMAL) #open the window which can be manually adjusted

# This threshold is used to filter noise, the contour area must be bigger than this to qualify as an actual contour.
noiseth = 500

while(1):
     
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
 
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
    else:            
       lower_range  = np.array([100,150,80])
       upper_range = np.array([140,255,255])
     
    mask = cv2.inRange(hsv, lower_range, upper_range)
     
    # Perform the morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
     
    # Find Contours in the frame.
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #(img, pick only outer contours, exclude all the excessiv pts save memory)
     
    # Make sure there is a contour present and also make sure its size is bigger than noise threshold.
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth: #max of contours wrt area
         
        # Grab the biggest contour with respect to area
        c = max(contours, key = cv2.contourArea) #Area represented in pixel.

        # Get bounding box coordinates around that contour
        x,y,w,h = cv2.boundingRect(c)
         
        # Draw that bounding box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2) #(frme, strt pt, end pt, color, thickness in pxls)
 
    cv2.imshow('image',frame)
     
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
 
cv2.destroyAllWindows()
cap.release()