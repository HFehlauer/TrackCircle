# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:25:48 2017
@author: Holger

script to find a circular shape in an image and track the relative movement of its center
the script requires:
    - at least one 8-bit image sequence as an .avi file with at least 10 images (the script will automatically
      analyze all .avi files in the defined folder)
    - a dark, circular-shaped object that stays in the field of view during the sequence
      this object needs to be the largest connected object with this intensity (not necessarily the only one)
    - no movement of the circle in the first 10 images, the center position of theses 10 images will be averaged
      and used as a reference point for the rest of the sequence
    - read and write access in the .avi file folder
the output is an .avi file: a copy of the input file with the outline of the detected circle and its center labelled
and a .csv file with the relative movement of the center of the circle
"""
        
import cv2                                                                     # for image/video i/o
import getpass                                                                 # for getting the user name - only necessary if folder is desktop on different pcs
import numpy as np
from skimage import measure                                                    # for finding the circle
import os                                                                      # for finding all .avi files in a folder

# you may edit this part
fileDir = 'C:/Users/'+getpass.getuser()+'/Desktop/CircleTracking/'             # folder of the .avi files that should be analysed - and also where the result .csv and .avi file ill be saved to
ptoDFac = 1                                                                    # pixel to distance conversion factor in ...

def findThreshold(image):
    ''' evaluates a threhold similar to the Otsu method:
            - splits the histogram in two parts
            - minimizes the sum of the variance of the two parts 
        input is an 8-bit bw image
        output is an 8-bit value - the threshold
    '''
    hist = cv2.calcHist([image],[0],None,[256],[0,256])                        # finds histogram
    hist_norm = hist.ravel()/hist.max()                                        # normalizes histogram
    Q = hist_norm.cumsum()                                                     # cumulates hisotgram
    bins = np.arange(256)                                                      # bins
    fn_min = np.inf                                                            # the minimum of the variance of the two histogram parts 
    thresh = -1                                                                # the threshold (id the same as with Otsu function)
    for i in range(1,256):                                                     # i is the value for splitting the histogram
        p1,p2 = np.hsplit(hist_norm,[i])                                       # the probabilities
        q1,q2 = Q[i],Q[255]-Q[i]                                               # cumulative sum of classes
        if q1 != 0 and q2 != 0:                                                # only evaluates the threshold if both parts of the histogram include values
            b1,b2 = np.hsplit(bins,[i])                                        # the weights
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2                         # the means
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2      # the variances
            fn = v1*q1 + v2*q2                                                 # calculates the minimization function
            if fn < fn_min:                                                    # if the minimum is smaller than any minimum before a new way to split the histogram was found
                fn_min = fn
                thresh = i
    return thresh-int(thresh/5)                                                # subtracts 1/5 from the Otsu threshold, this is hard-coded to compensate for the pipette intensity

def workonFile(f):                                                             # works on the defined .avi file
    ''' opens the image of the .avi file, does imageprocessing to find the circle,
        determines the midpoint of the circle and its movement along the main axis of movement
        input is an .avi file
        no output but generates an .avi file - a copy of the input file with an overlay of the detected circle and the midpoint
        and a .csv file with the distance traveled along the main axis
    '''
    cap = cv2.VideoCapture(fileDir+f)                                          # opens the video
    fourcc = cv2.VideoWriter_fourcc(*'DIB ')                                   # the codec for the output .avi file (might have to be changed according to installed codecs)
    out = cv2.VideoWriter(fileDir+'output'+f,fourcc, 20.0, (640,480))          # opens the videowriter for the output .avi file
    csvFile = open(fileDir+'result'+f.replace(' ', '')[:-4].upper()+'.csv','w')   # creates and opens a .csv file to save the result 
    i = 0                                                                      # counts the frames in the sequence
    xOpt,yOpt = [],[]                                                          # saves a list of the x,y coordinates of center of the circle                 
    while(i <= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1):                      # works on all frames in the sequence individually
        ret, frame = cap.read()                                                # reads the frame
        thresh = findThreshold(frame[:,:,0])                                   # finds the threshold with the findThreshold function
        ret1, img1 = cv2.threshold(frame[:,:,0],thresh,255,cv2.THRESH_BINARY_INV)   # threshold for the frame is automated 
        img1 = cv2.erode(img1, None, iterations=10)                            # erodes the frame 10 times
        img1 = cv2.dilate(img1, None, iterations=10)                           # dilates the frame 10 times
        label_img = measure.label(img1, connectivity=2, background=0)          # labels all connected regions in the frame
        mask = np.zeros(img1.shape, dtype="uint8")                             # creates a new frame that will become the mask
        maxPixels = 0                                                          # variable to measure the number of pixels in the largest labelled region of the frame
        k = 1                                                                  # variable for the number of labels, 0 is the background and therefore excluded
        for label in np.unique(label_img):                                     # goes through all labels
            ''' for all labels will detect the number of connected pixels and will determine the largest connected region
                and add this to the mask
            '''
            if label == 0:                                                     # label 0 is the background and excluded
                continue
            labelMask = np.zeros(img1.shape, dtype="uint8")                    # creates a frame for a mask with only the current label
            labelMask[label_img == label] = 255                                # sets the region of the current label to 255
            if cv2.countNonZero(labelMask)>= maxPixels:                        # determines whether the current region includes more pixels than the regions of all previous labels
                maxPixels = cv2.countNonZero(labelMask)                        # defines the highest number of pixels for any region so far
                largestObjectLabel = label                                     # largestObjectLabel is the label of the largest region
            k +=1     
            if k == len(np.unique(label_img)):                                 # if the size of all labeled regions have been compared
                labelMask = np.zeros(img1.shape, dtype="uint8")                # creates a new empty mask
                labelMask[label_img == largestObjectLabel] = 255               # sets the region of the largest object to 255
                mask = cv2.add(mask, labelMask)                                # adds this region to the mask frame
        regions = measure.regionprops(mask, cache=True)                        # measures a number of properties for all regions in the frame
        y0, x0 = regions[0].centroid                                           # positions of the center of a circle around the region
        r = regions[0].major_axis_length / 2.                                  # the radius of the circle
        try:
            cv2.circle(frame, (int(x0),int(y0)), int(r), 100)                  # tries to draw the circle into the frame
            cv2.drawMarker(frame, (int(x0),int(y0)),  int(r), cv2.MARKER_CROSS, 10, 1);  # tries a cross at the center of the circle into the frame
        except:
            pass
        xOpt.append(x0)                                                        # appends the x coordinate to the list of x coordinates
        yOpt.append(y0)                                                        # appends the y coordinate to the list of y coordinates
        out.write(frame)                                                       # writes the current frame (including the drawn circle and cross) into the output .avi file
        if i == int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1:
            xOpt, yOpt = xOpt-np.average(xOpt[0:9]), yOpt-np.average(yOpt[0:9])   # transforsm the origin to be at the average position of the circle in the first 10 images
            slope = np.polyfit(xOpt, yOpt, 1)[0]                               # slope of a linear regression through all new xOpt and yOpt pairs
            angle = np.arctan(slope)                                           # the slope as an angle
            xOptTrans = -(xOpt*np.cos(angle)+yOpt*np.sin(angle))               # transforms all xOpt coordinates to be along the axis of the slope
            for j in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1):
                csvFile.write(''+str(xOptTrans[j]*ptoDFac)+', \n')                     # writes the distances into a .csv file
        print('file: '+str(f)+ ', image number: '+str(i))
        i+=1                                                                   # counts through the frames of the sequence
    csvFile.close()                                                            # closes the .csv file
    out.release()                                                              # releases the output .avi file
    cap.release()                                                              # releases the input .avi file
    cv2.destroyAllWindows()                                                    # destroys all open opencv windows
    
aviFiles = []                                                                  # a list of the .avi files to work on
for dirpath, dirnames, filenames in os.walk(fileDir):
    for f in filenames:
        if os.path.splitext(f)[1] == '.avi':
            if not f.startswith('output'):
                aviFiles.append(os.path.join(f))
                
for f in aviFiles:                                                             # works on all .avi files
    workonFile(f)