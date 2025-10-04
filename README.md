
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import queue
import time
import random


# Speech synthesis class with macOS-safe queue and worker
class SpeechSynthesizer:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('volume', 1.0)
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
        self.speak_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def speak(self, text):
        if text and text.strip():
            self.speak_queue.put(text)

    def _worker(self):
        while True:
            text = self.speak_queue.get()
            if text == "__STOP__":
                break
            self.engine.say(text)
            self.engine.runAndWait()
            self.speak_queue.task_done()

    def stop(self):
        self.speak_queue.put("__STOP__")
        self.worker_thread.join()


# Hand gesture recognizer
class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.gestures = {
            'open_palm': 'Hello',
            'fist': 'Stop',
            'thumbs_up': 'Good',
            'peace': 'Peace'
        }

    def recognize(self, landmarks):
        if len(landmarks) != 21 * 3:
            return None
        lm = np.array(landmarks).reshape(21, 3)

        wrist = lm[0][:2]
        thumb_tip = lm[4][:2]
        index_tip = lm[8][:2]
        middle_tip = lm[12][:2]
        ring_tip = lm[16][:2]
        pinky_tip = lm[20][:2]

        def is_extended(tip, mcp):
            return tip[1] < mcp[1]

        thumb_ip = lm[3][:2]
        index_mcp = lm[5][:2]
        middle_mcp = lm[9][:2]
        ring_mcp = lm[13][:2]
        pinky_mcp = lm[17][:2]

        thumb_extended = is_extended(thumb_tip, thumb_ip)
        index_extended = is_extended(index_tip, index_mcp)
        middle_extended = is_extended(middle_tip, middle_mcp)
        ring_extended = is_extended(ring_tip, ring_mcp)
        pinky_extended = is_extended(pinky_tip, pinky_mcp)

        # Open palm: all fingers extended
        if all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return self.gestures['open_palm']

        # Fist: no fingers extended
        if not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return self.gestures['fist']

        # Thumbs up: thumb extended only
        if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return self.gestures['thumbs_up']

        # Peace: index and middle extended only
        if index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]):
            return self.gestures['peace']

        return None

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                gesture = self.recognize(landmarks)
                if gesture:
                    break
        return frame, gesture


# Face expression detector (placeholder)
class FaceExpressionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotions = ['Happy', 'Sad', 'Neutral', 'Surprise', 'Angry']

    def detect_expression(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        expressions = []
        for (x, y, w, h) in faces:
            emotion = random.choice(self.emotions)
            expressions.append((x, y, w, h, emotion))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame, expressions


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam. Check camera permissions and availability.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    speech = SpeechSynthesizer()
    hand_recognizer = HandGestureRecognizer()
    face_detector = FaceExpressionDetector()

    last_gesture = None
    gesture_cooldown = 1.5  # seconds
    last_gesture_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame, gesture = hand_recognizer.process_frame(frame)
            current_time = time.time()
            if gesture and (gesture != last_gesture or current_time - last_gesture_time > gesture_cooldown):
                print(f"Gesture recognized: {gesture}")
                speech.speak(gesture)
                last_gesture = gesture
                last_gesture_time = current_time
                cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            frame, expressions = face_detector.detect_expression(frame)
            for (x, y, w, h, emotion) in expressions:
                cv2.putText(frame, f"Face: {emotion}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                            2)
                print(f"Detected face expression: {emotion}")

            # Show the output frame in a window
            cv2.imshow("Virtual Camera Output", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        speech.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
