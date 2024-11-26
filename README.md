import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_hand_open(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    open_fingers = 0

    palm_y = landmarks[0].y

    for tip_id in tips_ids[1:]:  # Başparmak hariç diğer parmaklar
        if landmarks[tip_id].y < palm_y:  # Parmağın ucu avuçtan yukarıdaysa açıktır
            open_fingers += 1

    if landmarks[tips_ids[0]].x > landmarks[tips_ids[0] - 1].x:  # Başparmak sola açık
        open_fingers += 1

    return open_fingers >= 4  # En az 4 parmak açık olmalı

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    text = "CLOSE"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Parmakların açık/kapalı durumunu kontrol et
            if is_hand_open(hand_landmarks.landmark):
                text = "OPEN"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
