import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import math
import tensorflow as tf

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

# Define labels and translations
labels = ["C", "L", "U", "One", "Two", "Three", "Four", "Five", "Six", "Eight", "Nine", "Please", "Nice", "House"]
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
    "House": {"Hindi": "घर", "Gujarati": "ઘર"}
}

# Paths to downloaded font files
hindi_font_path = r"C:\Dewansh\Ml\Signlanguage\Fonts\NotoSansDevanagari-VariableFont_wdth,wght.ttf"
gujarati_font_path = r"C:\Dewansh\Ml\Signlanguage\Fonts\NotoSansGujarati-VariableFont_wdth,wght.ttf"

# Load fonts for Hindi and Gujarati
hindi_font = ImageFont.truetype(hindi_font_path, 32)
gujarati_font = ImageFont.truetype(gujarati_font_path, 32)

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
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                
                if index < len(labels):
                    label_text = labels[index]
                    hindi_text = translations.get(label_text, {}).get("Hindi", "अज्ञात")
                    gujarati_text = translations.get(label_text, {}).get("Gujarati", "અજ્ઞાત")
                else:
                    label_text = "Unknown"
                    hindi_text = "अज्ञात"
                    gujarati_text = "અજ્ઞાત"
            except Exception as e:
                print(f"Error during prediction: {e}")
                label_text = "Error"
                hindi_text = "त्रुटि"
                gujarati_text = "ત્રુટિ"

            # Overlay text using PIL for Hindi and Gujarati support
            img_pil = Image.fromarray(imgOutput)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x_min, y_min - 30), f"{label_text}", font=hindi_font, fill=(0, 0, 0))
            draw.text((x_min, y_min - 60), f"{hindi_text}", font=hindi_font, fill=(255, 0, 0))
            draw.text((x_min, y_min - 90), f"{gujarati_text}", font=gujarati_font, fill=(0, 255, 0))
            imgOutput = np.array(img_pil)

        cv2.rectangle(imgOutput, (x_min - 20, y_min - 20), 
                      (x_max + 20, y_max + 20), 
                      (0, 255, 0), 4)

    # Show the final image
    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
