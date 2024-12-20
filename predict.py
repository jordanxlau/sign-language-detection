import cv2
import keras
import tensorflow as tf
import numpy as np
from mediapipe import solutions

# List of labels
labels = {
    -1: 'no hand',
    0: 'call', 1: 'dislike', 2: 'fist', 3: 'four', 4: 'like', 5: 'mute',
    6: 'ok', 7: 'one', 8: 'palm', 9: 'peace', 10: 'peace_inverted', 11: 'rock',
    12: 'stop', 13: 'stop_inverted', 14: 'three', 15: 'three2', 16: 'two_up', 17: 'two_up_inverted'
}

# Find the best prediction
def best_prediction(predictions:np.array):
    # Search an np array of normalized predictions for the max value (most likely prediction)
    for j in range( len(predictions) ):
        max = 0
        for i in range( 18 ):
            max = i if predictions[j,max] < predictions[j,i] else max
    
    return max

try:
    # Rename mediapipe utilities
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    mp_hands = solutions.hands

    # Load the hand classification model
    model = keras.models.load_model("hand-gestures.keras")

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)
    success, image = cap.read()

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

        # Check for esc key
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Initialize a default prediction
        prediction = -1

        # Flip the image horizontally for a selfie-view image.
        image = cv2.flip(image, 1)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                for landmark in hand_landmarks.landmark:
                    # print([landmark.x, landmark.y, landmark.z])
                    hand.append(landmark.x)
                    hand.append(landmark.y)
                    hand.append(landmark.z)
                # Make a prediction
                hand = tf.constant([hand])
                predictions = model.predict(hand)
                prediction = best_prediction(predictions)

        # Draw a green text prediction
        cv2.putText(image, labels[prediction], (10, 100), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 200, 50), 2, cv2.LINE_AA)

        # Write to output video
        # result.write(image)

        # Re-image the image
        cv2.imshow('Sign Language Alphabet Detection', image)

        # Read next frame
        success, image = cap.read()

    # result.release()
    cap.release()
except Exception as e:
    print(e)