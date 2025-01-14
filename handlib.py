import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

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

            # Calculate distances between fingers (excluding thumb)
            index_middle_distance = calculate_distance(index_tip, middle_tip)
            middle_ring_distance = calculate_distance(middle_tip, ring_tip)
            ring_pinky_distance = calculate_distance(ring_tip, pinky_tip)

            # Calculate distances involving the thumb
            thumb_index_distance = calculate_distance(thumb_tip, index_tip)
            thumb_middle_distance = calculate_distance(thumb_tip, middle_tip)
            thumb_ring_distance = calculate_distance(thumb_tip, ring_tip)
            thumb_pinky_distance = calculate_distance(thumb_tip, pinky_tip)

            # Calculate distances involving the pinky
            pinky_index_distance = calculate_distance(pinky_tip, index_tip)
            pinky_middle_distance = calculate_distance(pinky_tip, middle_tip)

            # Define a small distance threshold for fingers to be considered close together
            small_distance_threshold = 0.05  # Adjust this value as needed
            # Define a larger distance threshold for the thumb to be considered apart
            large_distance_threshold = 0.1  # Adjust this value as needed

            if (
                index_middle_distance < small_distance_threshold
                and middle_ring_distance < small_distance_threshold
                and ring_pinky_distance < small_distance_threshold
                and thumb_index_distance > large_distance_threshold
                and thumb_middle_distance > large_distance_threshold
                and thumb_ring_distance > large_distance_threshold
                and thumb_pinky_distance > large_distance_threshold
                and thumb_tip.y < wrist.y
            ):
                print("Gesture: Thumbs Up")
            elif (
                index_middle_distance < small_distance_threshold
                and middle_ring_distance < small_distance_threshold
                and ring_pinky_distance < small_distance_threshold
                and thumb_index_distance > large_distance_threshold
                and thumb_middle_distance > large_distance_threshold
                and thumb_ring_distance > large_distance_threshold
                and thumb_pinky_distance > large_distance_threshold
                and thumb_tip.y > wrist.y
            ):
                print("Gesture: Thumbs Down")
            elif (
                thumb_tip.y < wrist.y
                and index_tip.y < wrist.y
                and middle_tip.y < wrist.y
                and ring_tip.y < wrist.y
                and pinky_tip.y < wrist.y
            ):
                print("Gesture: Stop Sign")
            elif (
                thumb_index_distance < small_distance_threshold
                and thumb_middle_distance < small_distance_threshold
                and thumb_ring_distance < small_distance_threshold
                and thumb_pinky_distance < small_distance_threshold
                and pinky_index_distance < small_distance_threshold
                and pinky_middle_distance < small_distance_threshold
                and index_middle_distance > large_distance_threshold
            ):
                print("Gesture: Peace Sign")
            else:
                print("Gesture: Unknown")

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
