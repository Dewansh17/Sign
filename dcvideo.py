import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Detect up to 2 hands
offset = 20
imgSize = 300
counter = 0

folder = "C:\\Dewansh\\Ml\\Signlanguage\\Data\\Sentences"

# Function to capture a sentence of sign language
def capture_sentence():
    sentence_start_time = time.time()
    sentence_images = []

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            # Process the detected hands for each word in the sentence
            for hand in hands:
                x, y, w, h = hand['bbox']

                # Create a white background image
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Crop the hand region from the original image
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                # Calculate aspect ratio and resize the cropped hand image
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                # Display the cropped and resized hand images
                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('Imagewhite', imgWhite)

                # Append the image to the sentence images list
                sentence_images.append(imgWhite)

            # Check for sentence completion (e.g., 'q' key press)
            key = cv2.waitKey(1)
            if key == ord("q"):
                sentence_end_time = time.time()
                sentence_duration = sentence_end_time - sentence_start_time

                # Save the sentence images as a single video or a sequence of images
                # Choose the appropriate method based on your requirements
                save_sentence_as_video(sentence_images)  # Or save_sentence_as_images(sentence_images)

                print(f"Sentence captured in {sentence_duration:.2f} seconds.")
                return sentence_duration

        # Display the original image with hand detection
        cv2.imshow('Image', img)

# Function to save the sentence images as a video
def save_sentence_as_video(sentence_images):
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify video codec
    video_path = f"{folder}/sentence_{time.time()}.mp4"
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (imgSize, imgSize))  # Adjust frame rate and size as needed

    # Write each image frame to the video
    for image in sentence_images:
        video_writer.write(image)

    # Release the video writer
    video_writer.release()

# Capture multiple sentences
num_sentences = int(input("Enter the number of sentences to capture: "))
sentence_durations = []
for i in range(num_sentences):
    print(f"Capturing sentence {i+1}")
    sentence_duration = capture_sentence()
    sentence_durations.append(sentence_duration)

# Print average sentence duration
average_duration = sum(sentence_durations) / len(sentence_durations)
print(f"Average sentence duration: {average_duration:.2f} seconds")

# Release resources
cap.release()
cv2.destroyAllWindows()