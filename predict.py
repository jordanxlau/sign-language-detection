import cv2
import keras
import tensorflow as tf
import numpy as np
from mediapipe import solutions

# List of labels (different from preprocessing.py)
labels = {
    -1: '',
    0: 'call', 1: 'dislike', 2: 'RESET', 3: 'four', 4: 'grabbing',
    5: 'grip', 6: 'heart', 7: 'heart', 8: 'praying', 9: 'like',
    10: 'little finger', 11: 'middle finger', 12: 'mute', 13: 'ok', 14: 'one',
    15: 'five', 16: 'two', 17: 'peace', 18: 'point', 19: 'rock',
    20: 'Bb', 21: 'stop', 22: 'Ww', 23: 'gun',
    24: 'three', 25: 'three', 26: 'Ll', 27: 'Uu', 28: 'two',
}

# Find the best prediction
def best_prediction(predictions:np.array):
    # Search an np array of normalized predictions for the max value (most likely prediction)
    max = 0
    for i in range( 29 ):
        max = i if predictions[0,max] < predictions[0,i] else max

    # Only output a prediction if the model is confident
    if predictions[0,max]<0.8:
        return -1
    else:
        return max

try:
    # Rename mediapipe
    mp_hands = solutions.hands

    # Define a mediapipe hand object
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.25)

    # List of key landmarks used to draw the boxes
    KEY_LANDMARKS = [mp_hands.HandLandmark.WRIST,
                     mp_hands.HandLandmark.THUMB_TIP,
                     mp_hands.HandLandmark.INDEX_FINGER_TIP,
                     mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                     mp_hands.HandLandmark.RING_FINGER_TIP,
                     mp_hands.HandLandmark.PINKY_TIP]

    # Load the hand classification model
    model = keras.models.load_model("hand-gestures-29.keras")

    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)
    success, image = cap.read()

    # Get info about VideoCapture
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define an output
    result = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'X264'), np.ceil(fps/3.5), (int(width), int(height)) , 0) # codec = method of compression
    
    # Initialize a previous prediction
    previous_prediction = -1
    message = " "

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
                # Find the top left and bottom right corner of the hand
                max_x, max_y, min_x, min_y = 0, 0, 1, 1
                for p in KEY_LANDMARKS:
                    x = hand_landmarks.landmark[p].x
                    y = hand_landmarks.landmark[p].y
                    max_x = x if x > max_x else max_x
                    max_y = y if y > max_y else max_y
                    min_x = x if x < min_x else min_x
                    min_y = y if y < min_y else min_y
                # Correct for the coordinate normalization done by mediapipe
                top_left = (int(max_x*width), int(max_y*height))
                bottom_right = (int(min_x*width),int(min_y*height))
                # Draw a rectangle around the hands
                cv2.rectangle(image, top_left, bottom_right, (0,0,255), 2)

                # Isolate all landmarks on the hand
                hand = []
                for landmark in hand_landmarks.landmark:
                    hand.append(landmark.x)
                    hand.append(landmark.y)
                    hand.append(landmark.z)

                # Make a prediction
                hand = tf.constant([hand])
                predictions = model.predict(hand)
                prediction = best_prediction(predictions)
                print(predictions)

                # Add to the message when the gesture changes
                if previous_prediction != prediction:
                    message = message + labels[prediction] + " " 
                if prediction == 2 or prediction == 32:
                    message = ""

                # Draw the string of gestures
                cv2.putText(image, message, top_left, cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        previous_prediction = prediction

        # Write to output video
        result.write(image)

        # Re-display the image
        cv2.imshow('Sign Language Alphabet Detection', image)

        # Read next frame
        success, image = cap.read()

    result.release()
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(e)