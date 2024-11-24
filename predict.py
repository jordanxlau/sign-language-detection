import cv2
import keras
import numpy as np
from mediapipe import solutions

# Find the best prediction
def best_prediction(predictions:np.array):
    # Search an np array of normalized predictions for the max value (most likely prediction)
    for j in range( len(predictions) ):
        max = 0
        for i in range( 26 ):
            max = i if predictions[j,max] < predictions[j,i] else max
    
    return max

# 
letters = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e',
    5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o',
    15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
    20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y',
    25: 'z', -1:''
}

# Rename mediapipe utilities
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
mp_hands = solutions.hands

# List of key landmarks used to draw the boxes
KEY_LANDMARKS = [mp_hands.HandLandmark.WRIST,
                mp_hands.HandLandmark.THUMB_CMC,
                mp_hands.HandLandmark.THUMB_MCP,
                mp_hands.HandLandmark.THUMB_IP,
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.INDEX_FINGER_DIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_PIP,
                mp_hands.HandLandmark.RING_FINGER_DIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_MCP,
                mp_hands.HandLandmark.PINKY_PIP,
                mp_hands.HandLandmark.PINKY_DIP,
                mp_hands.HandLandmark.PINKY_TIP]

# Load the hand classification model
model = keras.models.load_model("model-wholealphabet.keras")

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)
success, display = cap.read()

# Get info about VideoCapture
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

# Define an output
# result = cv2.VideoWriter("videos_partA_detection.mp4", cv2.VideoWriter_fourcc(*'X264'), fps, (int(width), int(height)) , 0) # codec = method of compression

# Define a mediapipe hand object
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Read from the video
while cap.isOpened():
    # Check if opened successfully
    if not success:
        continue # 'break' for videos, 'continue' for webcams

    # Initialize a default prediction
    prediction = -1

    # Flip the image horizontally for a selfie-view display.
    display = cv2.flip(display, 1)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    display.flags.writeable = False
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    results = hands.process(display)

    # Draw the hand annotations on the image.
    display.flags.writeable = True
    display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Find the top left and bottom right corner of the hand
            max_x, max_y, min_x, min_y = 0, 0, 1, 1
            for p in KEY_LANDMARKS:
                x = hand_landmarks.landmark[p].x
                y = hand_landmarks.landmark[p].y
                max_x = x if x > max_x else max_x
                max_y = y if y > max_y else max_y
                min_x = x if x < min_x else min_x
                min_y = y if y < min_y else min_y
            
            # Adjust for mediapipe coordinates
            max_y*=height
            max_x*=width
            min_y*=height
            min_x*=width
            # Crop the image in a square to just find the hand
            hand_image = display[int(min_y)-50:int(max_y)+50, int(min_x)-50:int(max_x)+50]

            # Prepare image prediction
            hand_image = cv2.resize(hand_image, (28, 28))
            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('Sign Language Alphabet Detection', hand_image)

            # Make a prediction
            hand_image = np.array([hand_image])
            predictions = model.predict(hand_image)
            prediction = best_prediction(predictions)

    # Draw a green text prediction
    cv2.putText(display, letters[prediction], (10, 100), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 200, 50), 2, cv2.LINE_AA)

    # Write to output video
    # result.write(image)

    # Re-display the image
    cv2.imshow('Sign Language Alphabet Detection', display)

    # Wait for esc key
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
    # Read next frame
    success, display = cap.read()

# result.release()
cap.release()