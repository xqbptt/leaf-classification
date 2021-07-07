import cv2
import numpy as np

orig_image = cv2.imread('level1.jpg',cv2.IMREAD_UNCHANGED) #contours will be loaded from this image
cntr_img = cv2.imread('level1.jpg',cv2.IMREAD_UNCHANGED) #contours will be drawn on this image (not really necessary but makes the code more readable)
label_img = cv2.imread('level1.jpg',cv2.IMREAD_UNCHANGED) #labels will be written on this image

img = cv2.medianBlur(orig_image,7) # blur the original image to reduce noise
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert it to grayscale
ret,thresh = cv2.threshold(gray,215,255,cv2.THRESH_BINARY_INV) # using 215 as threshold value worked well, since the image has very light yellow
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # finding contours with default options
valid_contours =[] # findcontours gives some contours which are very small like dots, so we'll remove these small invalid contours in the loop

yellow_lower = np.array([20, 100, 100]) #lower limit for yellow color in hsv
yellow_upper = np.array([30, 255, 255]) #upper limit for yellow color in hsv
green_lower = np.array([36,0,0]) # lower for green in hsv as well
green_upper = np.array([86,255,255]) # upper for green in hsv of course

for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)  #first find the bounding rectangle of the leaf
    if(w<20 or h<20): #if its area is too less, that means its not a leaf
        continue
    valid_contours += [contour] #append this contour to valid contour list
    cv2.rectangle(cntr_img,(x,y),(x+w,y+h),(200,200,200),1) #draw rectangles on cntr_img
    crop_img = orig_image[y:y+h,x:x+w,:] #crop out this section from the original image
    hsv_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV) #convert it to hsv for easy color detection
    mask_yellow = cv2.inRange(hsv_img, yellow_lower, yellow_upper) # find mask for yellow color
    mask_green = cv2.inRange(hsv_img, green_lower, green_upper) # find mask for green color
    yellow_ratio =(cv2.countNonZero(mask_yellow))/(crop_img.size/3) # ratio of yellow in image is equal to number of non zero pixels in mask to the ratio of total pixels
    green_ratio =(cv2.countNonZero(mask_green))/(crop_img.size/3) # same as above for green
    fresh = (green_ratio/(green_ratio+yellow_ratio)) #finds the ratio
    # if this value of fresh is greater than 0.7 then the leaf is fresh
    if fresh>0.7: 
        cv2.putText(label_img,'fresh[{perc:0.2f}%]'.format(perc = fresh*100), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0)) #puts label on label_img
    else:
        cv2.putText(label_img,'old[{perc:0.2f}%]'.format(perc = fresh*100), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255)) #same as above

cv2.drawContours(cntr_img, valid_contours, -1, (238,50,238), thickness = 2) #drawing contours on contour_img 

cv2.imwrite('output_labeled.jpg',label_img)

cv2.imwrite('output_contoured.jpg',cntr_img)
