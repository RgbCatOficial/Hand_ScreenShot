# Numpy is for the numpy-Array conversion
import numpy as np

# Pyautogui and imutils are responsable for taking the screenshot
import pyautogui
import imutils

# Cv2 and mediapipe Detect and capture the hand
import cv2
import mediapipe as mp

# Hand variables
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Video Capture in the main camera
cap = cv2.VideoCapture(0)

# Finger points (not the thumb) based in the mp hand catch system
finger_tips =[8, 12, 16, 20]
# Finger point (that is the thumb) based in the mp hand catch system
thumb_tip= 4

# In this loop, some actions are repeated until the code breaks:
# | 1. Catch the finger/thumb tips and landmarks
# | 2. Use the distance between them to check if the hand is closed
# | 3. If the hand is actually closed, the code screenshot automatically
while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    # This is the frame captured and with the tips and landmarks drawn by mp
    results = hands.process(img)

    # If the mp catched a hand in the camera, it goes to a for loop
    if results.multi_hand_landmarks:
        # Acess all landmark positions between points to check if they
        # |are at the certain position that makes the hand close
        print("Before hand detection loop")
        for hand_landmark in results.multi_hand_landmarks:
            print("In the loop")
            # Acessing the positions (based in the current hand_landmark in the loop, inside
            # | results.multi_hand_landmarks)

            # An empty list and a loop to save the values of the landmarks
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # An empty array to save if the landmarks indicates that the hand is closed({True}) or not ({False})    
            finger_fold_status = []
            for tip in finger_tips:
                # Catching the value of the distance between the points
                x,y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                # Drawing a circle to be visible in the pop-up
                cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)


                # Verify if the finger is bent by checking whether the initial value of the fingertip is less than
                # | the initial position of the finger which is the internal landmark for the index finger
                # The drawing color is changed by green if the condition is aproved, and the previous array
                # | for checking if each landmark is closed is appended with the value {True}
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x,y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                # If the distance between the points of the finger indicates that it is not bent, the array will be 
                # | appended with the value {False}, witch indicates that the hand musn`t be closed at this time 
                else:
                    finger_fold_status.append(False)

            # Is possible to view the landmark values about is bent or not in the terminal
            print(finger_fold_status)

            # In this if, is checked if all of the landmarks have the value of {True}, that
            # | indicated that the hand is closed.
            if all(finger_fold_status):
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), cv2.COLORGB2BGR)
                cv2.imwrite("in_memory_to_disk.png", image)
                pyautogui.screenshot("in_memory_to_disk.png")
                image = cv2.imread("in_memory_to_disk.png")
                cv2.imshow("Screen Capture", imutils.resize(image, width=600))
        print("After the loop")
    # This draw the landmarks in the Frame
    mp_draw.draw_landmarks(
        img,
        hand_landmark,
        mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec((0, 0, 255), 2, 2),
        mp_draw.DrawingSpec((0, 255, 0), 4, 2)
    )

    
    # Showing the frames
    cv2.imshow("Frame with drawing", img)
    cv2.waitKey(1)