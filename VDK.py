import cv2
import mediapipe as mp
import pygame
import time
import os  # Not sure if we need this yet, but leaving it in just in case

# Pygame audio setup - assuming sound files are in 'sounds/' folder
pygame.mixer.init()

# Loading up some drum samples - might swap these later
snare_sound = pygame.mixer.Sound("sounds/snare.wav")
kick_sound = pygame.mixer.Sound("sounds/kick.wav")
hihat_sound = pygame.mixer.Sound("sounds/hihat.wav")

# Setting up MediaPipe hands module
mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawer = mp.solutions.drawing_utils

# Open camera stream (0 = default webcam)
cam = cv2.VideoCapture(0)

# Track last time each sound was played to avoid spamming
last_hit_time = {"snare": 0, "kick": 0, "hihat": 0}
trigger_delay = 0.5  # half a sec cooldown between hits

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        print("Failed to read frame from camera")  # Human might leave this for debugging
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame so it's more intuitive
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_result = hand_tracker.process(frame_rgb)

    # Draw the drum "buttons" on screen
    cv2.rectangle(frame, (50, 100), (200, 250), (255, 0, 0), 2)
    cv2.putText(frame, "Snare", (75, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.rectangle(frame, (250, 100), (400, 250), (0, 255, 0), 2)
    cv2.putText(frame, "Kick", (280, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.rectangle(frame, (450, 100), (600, 250), (0, 0, 255), 2)
    cv2.putText(frame, "HiHat", (470, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Check for hand and landmarks
    if hand_result.multi_hand_landmarks:
        for hand in hand_result.multi_hand_landmarks:
            # Using index fingertip for hit detection
            index_tip = hand.landmark[8]
            x_pos = int(index_tip.x * width)
            y_pos = int(index_tip.y * height)

            # Draw fingertip circle
            cv2.circle(frame, (x_pos, y_pos), 10, (255, 255, 255), -1)

            current_time = time.time()

            # Hit detection based on rectangle zones
            if 50 < x_pos < 200 and 100 < y_pos < 250:
                if current_time - last_hit_time["snare"] > trigger_delay:
                    snare_sound.play()
                    last_hit_time["snare"] = current_time

            elif 250 < x_pos < 400 and 100 < y_pos < 250:
                if current_time - last_hit_time["kick"] > trigger_delay:
                    kick_sound.play()
                    last_hit_time["kick"] = current_time

            elif 450 < x_pos < 600 and 100 < y_pos < 250:
                if current_time - last_hit_time["hihat"] > trigger_delay:
                    hihat_sound.play()
                    last_hit_time["hihat"] = current_time

            # Draw landmarks on the hand
            drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Display the live feed
    cv2.imshow("Virtual Drum Kit", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
