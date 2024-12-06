import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
        self, mode=False, max_hands=2, min_detection_confi=0.5, min_track_confi=0.5
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.min_detection_confi = min_detection_confi
        self.min_track_confi = min_track_confi

        self.mpHands = mp.solutions.hands
        self.hand = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confi,
            min_tracking_confidence=self.min_track_confi,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hand.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_LM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, hand_LM, self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for idx, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (0, 0, 255), cv2.FILLED)

        return lm_list
