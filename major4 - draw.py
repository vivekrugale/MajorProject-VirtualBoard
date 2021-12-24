# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:38:25 2021

@author: VIVEK RUGLE
"""

#Major project drawing on board

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

#initialize the canvas on which we will draw
canvas = None

#initialize x1, y1 coordinates
x1, y1 = 0, 0

noiseth = 500

while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
     
    # Initialize the canvas as a black image of the same size as the frame.
    if canvas is None:
        canvas = np.zeros_like(frame)
 
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
    else:            
       lower_range  = np.array([100,150,80])
       upper_range = np.array([140,255,255])
     
    mask = cv2.inRange(hsv, lower_range, upper_range)
     
    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
     
    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
    # Make sure there is a contour present and also its size is bigger than the noise threshold.
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth: #max of contours wrt area

        c = max(contours, key = cv2.contourArea)   
        x2,y2,w,h = cv2.boundingRect(c)
         
        # If there were no previous points then save the detected x2,y2 coordinates as x1,y1.
        # This is true when we writing for the first time or when writing again when the pen had disappeared from view.
        
        draw = cv2.waitKey(1) & 0xFF
        if draw == ord('d'):
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
             
            else:
                # Draw the line on the canvas
                canvas = cv2.line(canvas, (x1,y1),(x2,y2), [0,255,0], 4)
        
        #Eraser functionality
        elif draw == ord('e'):
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
             
            else:
                # Draw the line on the canvas
                canvas = cv2.line(canvas, (x1,y1),(x2,y2), [0,0,0], 80)
         
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2
 
    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
     
    # Merge the canvas and the frame.
    frame = cv2.add(frame,canvas)
     
    # Optionally stack both frames and show it.
    stacked = np.hstack((canvas,frame))

    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))
 
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
         
    # When c is pressed clear the canvas
    if k == ord('c'):
        canvas = None
        
cv2.destroyAllWindows()
cap.release()