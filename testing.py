import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import math

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector for up to 2 hands
detector = HandDetector(maxHands=2)

# File paths for both models
model_path_1 = r"C:\Dewansh\Ml\Signlanguage\mode2\keras_model.h5"
labels_path_1 = r"C:\Dewansh\Ml\Signlanguage\mode2\labels.txt"
model_path_2 = r"C:\Dewansh\Ml\Signlanguage\mode3\alphabets_model.h5"  # New model path
labels_path_2 = r"C:\Dewansh\Ml\Signlanguage\mode3\alphabets_labels.txt"      # New label path

# Load both classifiers with error handling
try:
    classifier_1 = Classifier(model_path_1, labels_path_1)
    classifier_2 = Classifier(model_path_2, labels_path_2)
except Exception as e:
    print(f"Error loading classifiers: {e}")
    exit()

# Define labels and translations for both models
labels_1 = ["C", "L", "U", "One", "Two", "Three", "Four", "Five", "Six", "Eight", "Nine", "Please", "Nice", "House"]
labels_2 = ["A", "B"]  # Adjust these as per model 2's dataset

translations = {
    "C": {"Hindi": "सी", "Gujarati": "સી"},
    "L": {"Hindi": "एल", "Gujarati": "એલ"},
    "U": {"Hindi": "यू", "Gujarati": "યૂ"},
    "One": {"Hindi": "एक", "Gujarati": "એક"},
    "Two": {"Hindi": "दो", "Gujarati": "બે"},
    "Three": {"Hindi": "तीन", "Gujarati": "ત્રણ"},
    "Four": {"Hindi": "चार", "Gujarati": "ચાર"},
    "Five": {"Hindi": "पाँच", "Gujarati": "પાંચ"},
    "Six": {"Hindi": "छह", "Gujarati": "છ"},
    "Eight": {"Hindi": "आठ", "Gujarati": "આઠ"},
    "Nine": {"Hindi": "नौ", "Gujarati": "નવ"},
    "Please": {"Hindi": "कृपया", "Gujarati": "કૃપા"},
    "Nice": {"Hindi": "अच्छा", "Gujarati": "સારો"},
    "House": {"Hindi": "घर", "Gujarati": "ઘર"},
    "A": {"Hindi": "ए", "Gujarati": "એ"},
    "B": {"Hindi": "बी", "Gujarati": "બી"}
}

# Paths to downloaded font files
hindi_font_path = r"C:\Dewansh\Ml\Signlanguage\Fonts\NotoSansDevanagari-VariableFont_wdth,wght.ttf"
gujarati_font_path = r"C:\Dewansh\Ml\Signlanguage\Fonts\NotoSansGujarati-VariableFont_wdth,wght.ttf"

# Load fonts for Hindi and Gujarati
hindi_font = ImageFont.truetype(hindi_font_path, 32)
gujarati_font = ImageFont.truetype(gujarati_font_path, 32)

# Single-hand gesture labels
single_hand_gestures = ["C", "L", "U", "One", "Two", "Three", "Four", "Five", "Six", "Eight", "Nine", "Please", "Nice"]

# Confidence threshold for predictions (optional)
confidence_threshold = 0.7  # Adjust as needed

# Start the loop to process frames
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        imgSize = 300
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        predictions = []

        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]

            if imgCrop.size == 0:
                print("Invalid crop region, skipping.")
                continue

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

            try:
                # Use the appropriate classifier based on hand count
                if len(hands) == 1:
                    prediction, index = classifier_1.getPrediction(imgWhite, draw=False)
                    if index < len(labels_1) and labels_1[index] in single_hand_gestures:
                        predictions.append(labels_1[index])
                else:
                    prediction, index = classifier_2.getPrediction(imgWhite, draw=False)
                    if index < len(labels_2):
                        predictions.append(labels_2[index])

            except Exception as e:
                print(f"Error during prediction: {e}")

        # Majority voting on predictions if any, else "Unknown"
        if predictions:
            final_prediction = max(set(predictions), key=predictions.count)
        else:
            final_prediction = "Unknown"

        hindi_text = translations.get(final_prediction, {}).get("Hindi", "अज्ञात")
        gujarati_text = translations.get(final_prediction, {}).get("Gujarati", "અજ્ઞાત")

        img_pil = Image.fromarray(imgOutput)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x_min, y_min - 60), f"{final_prediction}", font=hindi_font, fill=(0, 0, 0))
        draw.text((x_min, y_min - 90), f"{hindi_text}", font=hindi_font, fill=(255, 0, 0))
        draw.text((x_min, y_min - 120), f"{gujarati_text}", font=gujarati_font, fill=(0, 255, 0))
        imgOutput = np.array(img_pil)

        cv2.rectangle(imgOutput, (x_min - 20, y_min - 20),
                      (x_max + 20, y_max + 20),
                      (0, 255, 0), 4)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
