import mediapipe as mp
import cv2

class Hand_Detector:
    def __init__(self):
        self.__hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7,
                model_complexity=1,
            )
        
    def __get_coordiantes(self, hand_landmarks, frame):
        h, w, c = frame.shape
        lm = hand_landmarks.landmark[9]
        cx, cy = int(lm.x * w), int(lm.y * h)
        return cx, cy
        
    def detect_hands(self, frame):
        detections = []
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.__hands.process(imgRGB)
        multi_hand_landmarks = results.multi_hand_landmarks
        if multi_hand_landmarks is None:
            return detections

        for hand_landmarks in multi_hand_landmarks:
            cx, cy = self.__get_coordiantes(hand_landmarks, frame)
            detections.append([cx, cy])
        
        return detections