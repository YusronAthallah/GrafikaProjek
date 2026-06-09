import math
import numpy as np

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def _count_fingers(hand_landmarks):

    lm = hand_landmarks.landmark

    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    WRIST = 0

    count = 0

    if lm[PINKY_MCP].x < lm[INDEX_MCP].x:
        if lm[THUMB_TIP].x > lm[THUMB_MCP].x:
            count += 1
    else:
        if lm[THUMB_TIP].x < lm[THUMB_MCP].x:
            count += 1

    finger_tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    finger_pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

    for tip, pip in zip(finger_tips, finger_pips):
        if lm[tip].y < lm[pip].y:
            count += 1

    return count


def detect_finger_number(hand_landmarks_list):
    if not hand_landmarks_list:
        return None

    total_fingers = 0
    for hand_landmarks in hand_landmarks_list:
        total_fingers += _count_fingers(hand_landmarks)

    if total_fingers == 0:
        return None

    total_fingers = min(total_fingers, 10)

    return str(total_fingers)
