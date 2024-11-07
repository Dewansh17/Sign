import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow up to 2 hands
offset = 20
imgSize = 300
counter = 0

folder = "C:\\Dewansh\\Ml\\Signlanguage\\Data\\B"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        # Initialize values for combined bounding box
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        # Loop through detected hands to update combined bounding box
        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Adjust the bounding box to fit both hands
        w_combined = x_max - x_min
        h_combined = y_max - y_min
        aspectRatio = h_combined / w_combined

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]
        
        if imgCrop.size == 0:
            print("Invalid crop region, skipping.")
            continue

        # Resize the cropped image to fit into imgWhite
        if aspectRatio > 1:
            k = imgSize / h_combined
            wCal = math.ceil(k * w_combined)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w_combined
            hCal = math.ceil(k * h_combined)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Display images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the main camera image
    cv2.imshow('Image', img)
    
    # Capture and save images with 'z' key
    key = cv2.waitKey(1)
    if key == ord("z"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Captured image {counter}")

cap.release()
cv2.destroyAllWindows()
