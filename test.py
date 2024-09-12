import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector for up to 2 hands
detector = HandDetector(maxHands=2)

# File paths
model_path = r"C:\Dewansh\Ml\Signlanguage\mode2\keras_model.h5"
labels_path = r"C:\Dewansh\Ml\Signlanguage\mode2\labels.txt"

# Load the classifier with error handling
try:
    classifier = Classifier(model_path, labels_path)
except Exception as e:
    print(f"Error loading classifier: {e}")
    exit()

offset = 20
imgSize = 300

# Define labels: numbers 0-9 and alphabets A-Z
labels = [
    "C", "L", "U", "One", "Two", "Three", "Four", "Five", "Six", "Eight", "Nine", "Please", "Nice", "House"
    # "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    # "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    # "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    # "U", "V", "W", "X", "Y", "Z"
]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        # Initialize variables for combining bounding boxes
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white background

        for hand in hands:  # Loop over detected hands
            x, y, w, h = hand['bbox']

            # Update combined bounding box coordinates
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

            # Crop the hand image
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Check if the crop is valid
            if imgCrop.size == 0:
                print("Invalid crop region, skipping.")
                continue

            aspectRatio = h / w

            # Resize and adjust for aspect ratio
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Predict the label
            try:
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                
                # Safeguard for index out of range
                if index < len(labels):
                    label_text = labels[index]
                else:
                    label_text = "Unknown"
            except Exception as e:
                print(f"Error during prediction: {e}")
                label_text = "Error"

            # Show the cropped hand images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        # Draw single bounding box for both hands
        cv2.rectangle(imgOutput, (x_min - offset, y_min - offset - 70), 
                      (x_min - offset + 400, y_min - offset + 60 - 50), 
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label_text, (x_min, y_min - 30), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x_min - offset, y_min - offset), 
                      (x_max + offset, y_max + offset), 
                      (0, 255, 0), 4)

    # Show the final image
    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
