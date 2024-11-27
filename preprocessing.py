from mediapipe import solutions
import cv2
import pandas as pd
import os
from tqdm import tqdm
import sys

# List of labels (different from predict.py)
labels = {
    0: 'call', 1: 'dislike', 2: 'fist', 3: 'four', 4: 'like',
    5: 'mute', 6: 'ok', 7: 'one', 8: 'palm', 9: 'peace',
    10: 'peace_inverted', 11: 'rock', 12: 'stop', 13: 'stop_inverted', 14: 'three',
    15: 'three2', 16: 'two_up', 17: 'two_up_inverted',
}

try:
    dataset_landmarked = [] # Processed dataset (each is a list with a label, then 21 3D landmark points)

    # Convert each image in the dataset to hand landmarks using mediapipe
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    mp_hands = solutions.hands

    # Define a mediapipe Hands object
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4)

    # Read data for each class
    for i in range(18):
        label = labels[i]
        folder = os.listdir("hagrid_dataset_512/" + label)

        # Read all images in the folder for class i
        for j in tqdm(range(min(len(folder),10000)), desc=label, file=sys.stdout): # progress bar code from: https://www.geeksforgeeks.org/progress-bars-in-python/ with file=sys.stdout suggestion from ChatGPT
            image_name = folder[j]
            
            image = cv2.imread("hagrid_dataset_512/" + label + "/" + image_name)

            # Uncomment this code to display each image, for testing purposes
            # cv2.imshow("Trace Image", image)
            # cv2.waitKey(0)

            # Flip image for "selfie" view
            image = cv2.flip(image, 1)
            
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image) # Process any hands in the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # If no hands detected, move to next image
            if not results.multi_hand_landmarks:
                continue

            # Get landmarks from first hand detected 
            hand_landmarks = results.multi_hand_landmarks[0]

            # Uncomment this code to display each image's hand landmarks, for testing purposes
            # Trace by showing image
            # annotated_image = image.copy()
            # mp_drawing.draw_landmarks(
            #     annotated_image,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())
            # cv2.imshow("Trace Annotated Image", annotated_image)
            # cv2.waitKey(0)
            
            # Group all landmark points into one "hand"
            hand = [i]
            for landmark in hand_landmarks.landmark:
                hand.append(landmark.x)
                hand.append(landmark.y)
                hand.append(landmark.z)

            # Add current hand example to processed data set
            dataset_landmarked.append(hand)

    # Convert to DataFrame
    dataset_landmarked = pd.DataFrame(dataset_landmarked)

    # Save as .csv files
    dataset_landmarked.to_csv("dataset_landmarked.csv", index=False)

except Exception as e:
    print("Exception:", e)