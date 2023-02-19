# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:43:39 2022

@author: VIVEK RUGLE
"""

import cv2
import numpy as np
import time

load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')
    
def nothing(x):
    pass

# Initializing the webcam feed.
cap = cv2.VideoCapture(0)
cap.set(5,1280)
cap.set(4,720)

# Create a window named trackbars.
cv2.namedWindow("Trackbars")
 
# create 6 trackbars that will control the lower and upper range of H,S and V channels. The Arguments are like this: Name of trackbar,
# window name, range,callback function. For Hue the range is 0-179 and  for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Load these 2 images and resize them to the same size.
pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.png',1), (50, 50))

kernel = np.ones((5,5),np.uint8)

# This is the canvas on which we will draw upon
canvas = None

# Create a background subtractor Object
backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

# This threshold determines the amount of disruption in the background.
background_threshold = 600

# A variable which tells you if you're using a pen or an eraser.
switch = 'Pen'

# With this variable we will monitor the time between previous switch.
last_switch = time.time()

# Initilize x1,y1 points
x1,y1=0,0

# Threshold for noise
noiseth = 800

# A variable which tells when to clear canvas
clear = False
  
while True:
     
    # Start reading the webcam feed frame by frame.
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally
    frame = cv2.flip( frame, 1 )
    
        # Initilize the canvas as a black image
    if canvas is None:
        canvas = np.zeros_like(frame)
        
    # Take the top left of the frame and apply the background subtractor there    
    top_left = frame[0: 50, 0: 50]
    fgmask = backgroundobject.apply(top_left)
    
    # Note the number of pixels that are white, this is the level of disruption.
    switch_thresh = np.sum(fgmask==255)
    
    # If the disruption is greater than background threshold and there has 
    # been some time after the previous switch then you. can change the  object type.
    if switch_thresh>background_threshold and (time.time()-last_switch) > 1:

        # Save the time of the switch. 
        last_switch = time.time()
        
        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'

     
    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    # Get the new values of the trackbar in real time as the user changes them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    #l_h = 100
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    #l_s = 150
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    #l_v = 80
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    #u_h = 140
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    #u_s = 255
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    #u_v = 255
  
    # If you're reading from memory then load the upper and lower ranges 
    # from there
    if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:             
       lower_range  = np.array([100,150,80])
       upper_range = np.array([140,255,255])
     
    # Filter the image and get the binary mask, where white represents your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
  
    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
     
    # stack the mask orginal frame
    stacked1 = np.hstack((mask_3,frame))
     
    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars',cv2.resize(stacked1,None,fx=0.5,fy=0.5)) # (original, desired size, scale along x axis, y)
    
    
    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    
    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also it size is bigger than noise threshold.
    if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
                
        c = max(contours, key = cv2.contourArea)    
        x2,y2,w,h = cv2.boundingRect(c)
        
        # Get the area of the contour
        area = cv2.contourArea(c)
        
        draw = cv2.waitKey(1) & 0xFF
        
        #Thickness
        t = 3
        
        #BLUE
        if draw == ord('b'):
        
            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,0,0], t)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
        
        #GREEN
        elif draw == ord('g'):
        
            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), [0,255,0], t)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
                    
        #RED
        elif draw == ord('r'):
        
            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), [0,0,255], t)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
                    
        #YELLOW
        elif draw == ord('y'):
        
            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), [0,255,255], t)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
                    
        #PINK
        elif draw == ord('p'):
        
            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,0,255], t)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
        
        #CYAN
        elif draw == ord('q'):
        
            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,255,0], t)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
        
        #WHITE
        elif draw == ord('w'):
        
            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
                
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), [255,255,255], t)
                    
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
                    
        #Eraser functionality
        elif draw == ord('e'):
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
             
            else:
                # Draw the line on the canvas
                cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
            
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2

    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    

    # Switch the images depending upon what we're using, pen or eraser.
    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0: 50, 0: 50] = eraser_img
    else:
        cv2.circle(frame, (x1, y1), 5, (255,255,255), -1)
        frame[0: 50, 0: 50] = pen_img
    
    # Merge the canvas and the frame.
    frame = cv2.add(frame,canvas)
     
    # Optionally stack both frames and show it.
    stacked = np.hstack((canvas,frame))

    cv2.imshow('Virtual Board', stacked)
    
    key = cv2.waitKey(1)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
    # When c is pressed clear the canvas
    if k == ord('c'):
        canvas = None
     
    # If the user presses `s` then print this array.
    if key == ord('s'):
         
        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)
         
        # Also save this array as penval.npy
        np.save('penval',thearray)
        break
    
cv2.destroyAllWindows()
cap.release()