from mediapipe import solutions
import cv2
import pandas as pd
import os

# Dataset from: https://github.com/hukenovs/hagrid/tree/Hagrid_v2?tab=readme-ov-file
labels = {
    0: 'call', 1: 'dislike', 2: 'fist', 3: 'four', 4: 'like'
}

try:
    X = [] # Processed images (each is a list of 21 3D landmark points)
    y = [] # Class labels

    # Convert each image in the dataset to hand landmarks using mediapipe
    mp_drawing = solutions.drawing_utils
    mp_drawing_styles = solutions.drawing_styles
    mp_hands = solutions.hands

    # Define a mediapipe Hands object
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.25)

    # Read data for each class
    for i in range(len(labels)):
        label = labels[i]
        folder = os.listdir("hagrid_dataset_512/" + label)
        print("label:", label)
        counter = 0

        # Read all images in the folder for class i
        for image_name in folder:
            # Only process the first 1000 examples
            if counter > 1000:
                break
            else:
                counter+=1
            
            image = cv2.imread("hagrid_dataset_512/" + label + "/" + image_name)

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
            hand = []
            for landmark in hand_landmarks.landmark:
                # print([landmark.x, landmark.y, landmark.z])
                hand.append(landmark.x)
                hand.append(landmark.y)
                hand.append(landmark.z)

            # Add current hand example to processed data set
            X.append(hand)
            y.append(i)
            print("\n\n\n\nhand:", hand)

    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    print(X)
    print(y)

    X.to_csv("X.csv", index=False)
    y.to_csv("y.csv", index=False)

except Exception as e:
    print("Exception:", e)