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

# Initialize hand detector
detector = HandDetector(maxHands=1)

# File paths
model_path = r"C:\Dewansh\Ml\Signlanguage\Model\keras_model.h5"
labels_path = r"C:\Dewansh\Ml\Signlanguage\Model\labels.txt"

# Load the classifier with error handling
try:
    classifier = Classifier(model_path, labels_path)
except Exception as e:
    print(f"Error loading classifier: {e}")
    exit()

offset = 20
imgSize = 300
counter = 0

# Define labels
labels = ["Hello" ,"Thankyou", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image for resizing
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand image
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Predict the label
        try:
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label_text = labels[index]
        except Exception as e:
            print(f"Error during prediction: {e}")
            label_text = "Error"

        # Draw results on the image
        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label_text, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Show the images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
