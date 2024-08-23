# this code is not the main code ..this code is for data collection
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import time

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)   
offset = 20
imgSize = 300
counter = 0

folder="C:\Dewansh\Ml\Signlanguage\Data\Thankyou"

# coding for data collection

while True:
    success , img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape=imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            # we are woking with height in if condition and we are working on weight on else condition
            k=imgSize / h 
            # calculate weight 
            wCal = math.ceil(k * w )   
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            # weight gap
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[ :, wGap: wCal+wGap] = imgResize

        else:
            k=imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop',imgCrop)
        cv2.imshow('Imagewhite',imgWhite)
    # to open the camera we have to stay in the loop
    cv2.imshow('Image', img)
    # key is the keyboard pe jo key hogi jisse press karne par data ko collect kar payegae for example in this case we have written 's' to capture the data 
    key = cv2. waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg' , imgWhite)
        print(counter)










