import cv2
import numpy as np

yellow_lower = np.array([20, 100, 100]) #lower limit for yellow color in hsv
yellow_upper = np.array([30, 255, 255]) #upper limit for yellow color in hsv
green_lower = np.array([36,0,0]) # lower for green in hsv as well
green_upper = np.array([86,255,255]) # upper for green in hsv of course

def get_green(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #convert it to hsv for easy color detection
    mask_yellow = cv2.inRange(hsv_img, yellow_lower, yellow_upper) # find mask for yellow color
    mask_green = cv2.inRange(hsv_img, green_lower, green_upper) # find mask for green color
    yellow_ratio =(cv2.countNonZero(mask_yellow))/(img.size/3) # ratio of yellow in image is equal to number of non zero pixels in mask to the ratio of total pixels
    green_ratio =(cv2.countNonZero(mask_green))/(img.size/3) # same as above for green
    return (green_ratio/(green_ratio+yellow_ratio))


img_maple = cv2.imread('neemleafcorrect.jpg')
test_maple = cv2.imread('neemleaves.jpg')
img_maple_gray = cv2.cvtColor(img_maple,cv2.COLOR_BGR2GRAY)
orig_green = get_green(img_maple)
print(orig_green*100)

orb = cv2.ORB_create(nfeatures = 250)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

kp_o, des_o = orb.detectAndCompute(img_maple_gray,None)

imgkp_o = cv2.drawKeypoints(img_maple,kp_o,None)


gray_test = cv2.cvtColor(test_maple,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_test,230,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

valid_contours =[]
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)  #first find the bounding rectangle of the leaf
    if(w<100 and h<100): #if its area is too less, that means its not a leaf
        continue
    valid_contours += [contour] #append this contour to valid contour list
    crop_img_gray = gray_test[y:y+h,x:x+w]
    crop_img = test_maple[y:y+h,x:x+w]
    test_green = get_green(crop_img)
    kp_t, des_t = orb.detectAndCompute(crop_img_gray,None)
    #imgkp_t = cv2.drawKeypoints(crop_img_gray,kp_t,None)
    try:
        matches = bf.knnMatch(des_o,des_t,k=2)
    except:
        continue
    goodmatches = 0
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodmatches = goodmatches + 1
    similarity = 30*(goodmatches/15) + 70*test_green
    cv2.putText(test_maple,"similarity: {s:0.2f}".format(s=similarity) , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0)) #puts label on label_img
    cv2.rectangle(test_maple,(x,y),(x+w,y+h),(200,200,200),1) #draw rectangles on cntr_img
print(len(valid_contours))
cv2.drawContours(test_maple, valid_contours, -1, (238,50,238), thickness = 2) #drawing contours on contour_img 

#cv2.imshow('yo',gray_test)
print(len(valid_contours))
cv2.imwrite('level2outneem.jpg',test_maple)
