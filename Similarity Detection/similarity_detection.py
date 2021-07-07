# in this approach two parameters have been used to find out similarity
# the traditional approach to finding similar objects uses feature detection and matching
# in addition to that the green content of the images has also been compared

import cv2
import numpy as np

yellow_lower = np.array([20, 100, 100]) #lower limit for yellow color in hsv
yellow_upper = np.array([30, 255, 255]) #upper limit for yellow color in hsv
green_lower = np.array([36,0,0]) # lower for green in hsv as well
green_upper = np.array([86,255,255]) # upper for green in hsv of course
#in this function dark green color can also be detected for much more accurate results
def get_green(img): # this function finds the green component, (this can be optimised by testing a bunch of different return values)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #convert it to hsv for easy color detection
    mask_yellow = cv2.inRange(hsv_img, yellow_lower, yellow_upper) # find mask for yellow color
    mask_green = cv2.inRange(hsv_img, green_lower, green_upper) # find mask for green color
    yellow_ratio =(cv2.countNonZero(mask_yellow))/(img.size/3) # ratio of yellow in image is equal to number of non zero pixels in mask to the ratio of total pixels
    green_ratio =(cv2.countNonZero(mask_green))/(img.size/3) # same as above for green
    return green_ratio/(green_ratio+yellow_ratio)

# FOR MAPLE LEAF----------------------------------------------------------------------------------
img_maple = cv2.imread('mapleleafcorrect.jpg')
test_maple = cv2.imread('mapleleaves.jpg')
img_maple_gray = cv2.cvtColor(img_maple,cv2.COLOR_BGR2GRAY) #convert to gray for feature detection

#find the contours of test image in order to separate out individual leaves
gray_test = cv2.cvtColor(test_maple,cv2.COLOR_BGR2GRAY) #same as above
ret,thresh = cv2.threshold(gray_test,230,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
valid_contours =[] #dictionary to store valid contours
#---------------------------------------

orb = cv2.ORB_create(nfeatures = 250)  #for feature detection
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

kp_o, des_o = orb.detectAndCompute(img_maple_gray,None) #find features of perfect leaf image
orig_green = get_green(img_maple) # green component of original image, this is close to 100

for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)  #first find the bounding rectangle of the leaf
    if(w<100 or h<100): #if its area is too less, that means its not a leaf
        continue
    valid_contours += [contour] #append this contour to valid contour list
    crop_img_gray = gray_test[y:y+h,x:x+w] #gray image for feature detection
    crop_img = test_maple[y:y+h,x:x+w] #color image for finding green component
    test_green = get_green(crop_img) #find green component of this leaf
    kp_t, des_t = orb.detectAndCompute(crop_img_gray,None)
    matches = bf.knnMatch(des_o,des_t,k=2)
    goodmatches = 0   #finds the number of goodmatches, more goodmatches means more similar
    for m,n in matches:
        if m.distance < 0.75*n.distance: #choice of 0.75 is arbitrary, it can be optimised for better results
            goodmatches = goodmatches + 1
    similarity = 50*(goodmatches/30) + 50*test_green # i divided the goodmatches by 30 which was about the max number of goodmatchs for any leaf
    #then i took 50-50 contribution of both feature similarity and color similarity to give final prediction
    #the choice 50-50 can be optimised, i took 50-50 by intuition
    cv2.putText(test_maple,"similarity: {s:0.2f}".format(s=similarity) , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0)) #put label on final image
    cv2.rectangle(test_maple,(x,y),(x+w,y+h),(200,200,200),1) #draws rectangle around the leaf

#cv2.drawContours(test_maple, valid_contours, -1, (238,50,238), thickness = 2) #drawing contours on contour_img #can be uncommented if contours need to be drawn
cv2.imwrite('level2out.jpg',test_maple)
#code for maple leaf ends--------------------------------------------------------------------------------
#this can be converted into a function which takes an image as input so that the code doesnt have to be repeated for neem leaf, but for now... :)

#FOR NEEM LEAF---------------------------------------------------------------------------
img_neem = cv2.imread('neemleafcorrect.jpg')
test_neem = cv2.imread('neemleaves.jpg')
img_neem_gray = cv2.cvtColor(img_neem,cv2.COLOR_BGR2GRAY)
orig_green = get_green(img_neem)

orb = cv2.ORB_create(nfeatures = 500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2)

kp_o, des_o = orb.detectAndCompute(img_neem,None)
gray_test = cv2.cvtColor(test_neem,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_test,230,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

valid_contours =[]

for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)  #first find the bounding rectangle of the leaf
    if(w<100 and h<100): #if its area is too less, that means its not a leaf
        continue
    valid_contours += [contour] #append this contour to valid contour list
    crop_img_gray = gray_test[y:y+h,x:x+w]
    crop_img = test_neem[y:y+h,x:x+w]
    test_green = get_green(crop_img)
    kp_t, des_t = orb.detectAndCompute(crop_img,None)
    try:
        matches = bf.knnMatch(des_o,des_t,k=2)
    except:
        continue
    goodmatches = 0
    for m,n in matches:
        if m.distance < 0.80*n.distance:
            goodmatches = goodmatches + 1
    if(goodmatches>50):
        similarity = 100
    else:
        similarity = 40*(goodmatches/30) + 60*test_green
    cv2.putText(test_neem,"similarity: {s:0.2f}".format(s=similarity) , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0)) #puts label on label_img
    cv2.rectangle(test_neem,(x,y),(x+w,y+h),(200,200,200),1) #draw rectangles on cntr_img
    
#cv2.drawContours(test_maple, valid_contours, -1, (238,50,238), thickness = 2) #drawing contours on contour_img 

cv2.imwrite('level2outneem.jpg',test_neem)
