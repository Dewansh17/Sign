import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf
from collections import Counter

# Print TensorFlow and Keras versions
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector for up to 2 hands
detector = HandDetector(maxHands=2)

# File paths
model_path = r"C:\Dewansh\Ml\Signlanguage\Trial\keras_model.h5"
labels_path = r"C:\Dewansh\Ml\Signlanguage\Trial\labels.txt"
try:
    classifier = Classifier(model_path, labels_path)
except Exception as e:
    print(f"Error loading classifier: {e}")
    exit()

# Constants
offset = 20
imgSize = 300
buffer_size = 30  # Number of frames for majority voting buffer
frame_buffer = []  # Buffer to store recent predictions

# Define labels for gestures/phrases
labels = [
    "Hello, how are you", "C", "L", "Nice", "Thank you", "Please", "Yes", "House"
]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        # Find the overall bounding box that includes both hands
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Crop combined bounding box area for both hands
        imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]
        if imgCrop.size == 0:
            continue

        # Resize and fit to square canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        h, w, _ = imgCrop.shape
        aspectRatio = h / w

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

        # Predict the gesture
        try:
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if index < len(labels):
                frame_buffer.append(labels[index])
            else:
                frame_buffer.append("Unknown")

            # Limit the buffer size
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            # Get the most common label from the buffer (majority vote)
            final_label = Counter(frame_buffer).most_common(1)[0][0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            final_label = "Error"

        # Display label on output image
        cv2.putText(imgOutput, final_label, (x_min, y_min - 30), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(imgOutput, (x_min - offset, y_min - offset), 
                      (x_max + offset, y_max + offset), (0, 255, 0), 4)

        # Show cropped image
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # Display the output
    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
