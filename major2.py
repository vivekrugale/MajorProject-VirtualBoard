# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 00:00:31 2021

@author: VIVEK RUGLE
"""
#Colour filteration

import cv2
import numpy as np

load_from_disk = True
 
if load_from_disk:
    penval = np.load('penval.npy')

#penval = [[100, 150, 80], [140, 255, 255]]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
 
# Creating A 5x5 kernel for morphological operations
kernel = np.ones((5,5),np.uint8)
 
while(1):
     
    ret, frame = cap.read() #ret = true/false if frame is present
    if not ret:
        break
         
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
     
    # Perform the morphological opn.s to get rid of the noise Erosion Eats away the white part while dilation expands it.
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
 
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    stacked = np.hstack((mask_3,frame))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
     
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()