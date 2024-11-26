import cv2
import keras
import tensorflow as tf
import numpy as np
from mediapipe import solutions

# Find the best prediction
def best_prediction(predictions:np.array):
    # Search an np array of normalized predictions for the max value (most likely prediction)
    for j in range( len(predictions) ):
        max = 0
        for i in range( 5 ):
            max = i if predictions[j,max] < predictions[j,i] else max
    
    return max

# List of labels 
labels = {
    -1:'No Hand Detected', 0: 'call', 1: 'dislike', 2: 'fist', 3: 'four', 4: 'like'
}

try:
        
    # Rename mediapipe utilities
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    mp_hands = solutions.hands

    # Load the hand classification model
    model = keras.models.load_model("model-fivesymbols.keras")

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
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.25)

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
            print("HEYO")
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                for landmark in hand_landmarks.landmark:
                    # print([landmark.x, landmark.y, landmark.z])
                    hand.append(landmark.x)
                    hand.append(landmark.y)
                    hand.append(landmark.z)
                # Make a prediction
                print(hand)
                hand = tf.constant([hand])
                print(hand.shape)
                predictions = model.predict(hand)
                print("DID WE MAKE IT HERE")
                prediction = best_prediction(predictions)

        # Draw a green text prediction
        cv2.putText(display, labels[prediction], (10, 100), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 200, 50), 2, cv2.LINE_AA)

        # Write to output video
        # result.write(image)

        # Re-display the image
        cv2.imshow('Sign Language Alphabet Detection', display)
        
        # Check for esc key
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Read next frame
        success, display = cap.read()

    # result.release()
    cap.release()
except Exception as e:
    print(e)