import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame from the camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

       
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            if (
                thumb_tip.y < wrist.y
                and index_tip.y < wrist.y
                and middle_tip.y < wrist.y
                and ring_tip.y < wrist.y
                and pinky_tip.y < wrist.y
            ):
                print("Gesture: Stop Sign")
            elif (
                thumb_tip.y < wrist.y
                and index_tip.y > wrist.y
                and middle_tip.y > wrist.y
                and ring_tip.y > wrist.y
                and pinky_tip.y > wrist.y
            ):
                print("Gesture: Thumbs Up")
            elif (
                thumb_tip.y > wrist.y
                and index_tip.y > wrist.y
                and middle_tip.y > wrist.y
                and ring_tip.y > wrist.y
                and pinky_tip.y > wrist.y
            ):
                print("Gesture: Thumbs Down")
            elif (
                index_tip.y < wrist.y
                and middle_tip.y < wrist.y
                and ring_tip.y > wrist.y
                and pinky_tip.y > wrist.y
            ):
                print("Gesture: Peace Sign")
            else:
                print("Gesture: Unknown")

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
