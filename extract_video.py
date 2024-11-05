import cv2
import os

# Configuration
video_path = "C:\\Dewansh\\Ml\\Signlanguage\\Data\\Hello,How are you\\sentence_1729774112.2474425.mp4"  # Path to your video file
output_folder = "C:\\Dewansh\\Ml\\Signlanguage\\Data\\Frames"  # Folder to save the frames as images
phrase_label = "hello_how_are_you"  # Label for the phrase (replace commas for file naming)

# Frame extraction settings
frame_interval = 10  # Extract every 10th frame for a balanced dataset

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)  # Create the main output folder

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

frame_count = 0
extracted_count = 0

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read.")
        break  # Exit the loop if no more frames are available

    # Save every nth frame based on the frame interval
    if frame_count % frame_interval == 0:
        output_path = os.path.join(output_folder, f"{phrase_label}_{extracted_count}.jpg")
        cv2.imwrite(output_path, frame)
        extracted_count += 1
        print(f"Saved frame {extracted_count} to {output_path}")

    frame_count += 1

# Release the video capture object
cap.release()
print("Frame extraction complete.")
