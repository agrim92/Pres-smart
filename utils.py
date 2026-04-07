"""Shared utilities for gesture detection and OpenCV overlays."""

from __future__ import annotations

import time
from typing import Dict

import cv2


FINGER_TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_PIP_IDS = {
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

HAND_CONNECTIONS = (
    (0, 1),
    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
)


def get_landmarks(hand_landmarks):
    """Normalize MediaPipe landmark containers across API versions."""
    if hasattr(hand_landmarks, "landmark"):
        return hand_landmarks.landmark
    return hand_landmarks


def euclidean_distance(point_a, point_b) -> float:
    """Return the 2D Euclidean distance between two MediaPipe landmarks."""
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    return (dx * dx + dy * dy) ** 0.5


def get_finger_states(hand_landmarks, handedness_label: str) -> Dict[str, bool]:
    """Return a boolean map describing which fingers appear extended."""
    landmarks = get_landmarks(hand_landmarks)

    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    palm_width = max(euclidean_distance(index_mcp, pinky_mcp), 0.01)

    thumb_tip = landmarks[FINGER_TIP_IDS["thumb"]]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    thumb_extension = euclidean_distance(thumb_tip, thumb_mcp) / palm_width

    if handedness_label == "Right":
        thumb_open = thumb_tip.x < thumb_ip.x and thumb_extension > 0.95
    else:
        thumb_open = thumb_tip.x > thumb_ip.x and thumb_extension > 0.95

    states = {"thumb": thumb_open}
    for finger_name, tip_id in FINGER_TIP_IDS.items():
        if finger_name == "thumb":
            continue
        pip_id = FINGER_PIP_IDS[finger_name]
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        states[finger_name] = tip.y < pip.y

    return states


def count_extended_fingers(finger_states: Dict[str, bool]) -> int:
    """Count how many fingers are extended."""
    return sum(1 for is_extended in finger_states.values() if is_extended)


def draw_hand_landmarks(frame, hand_landmarks) -> None:
    """Render hand landmarks, connections, and a bounding box."""
    landmarks = get_landmarks(hand_landmarks)
    frame_height, frame_width = frame.shape[:2]
    xs = [int(landmark.x * frame_width) for landmark in landmarks]
    ys = [int(landmark.y * frame_height) for landmark in landmarks]
    padding = 18
    left = max(min(xs) - padding, 0)
    right = min(max(xs) + padding, frame_width - 1)
    top = max(min(ys) - padding, 0)
    bottom = min(max(ys) + padding, frame_height - 1)

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 140), 2, cv2.LINE_AA)

    for start_idx, end_idx in HAND_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * frame_width), int(start.y * frame_height))
        end_point = (int(end.x * frame_width), int(end.y * frame_height))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2, cv2.LINE_AA)

    for landmark in landmarks:
        center = (int(landmark.x * frame_width), int(landmark.y * frame_height))
        cv2.circle(frame, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 6, (0, 200, 0), 1, cv2.LINE_AA)


def draw_overlay(
    frame,
    interaction_mode: bool,
    gesture_text: str,
    debug_text: str = "",
    fps: float = 0.0,
) -> None:
    """Draw a semi-transparent status panel on top of the webcam frame."""
    panel_color = (40, 160, 60) if interaction_mode else (50, 70, 210)
    status_text = "Interaction Mode ON" if interaction_mode else "Interaction Mode OFF"
    indicator_color = (70, 220, 90) if interaction_mode else (70, 70, 230)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, 142), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (10, 10), (520, 142), panel_color, 2)
    cv2.circle(frame, (34, 38), 10, indicator_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (34, 38), 12, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        status_text,
        (56, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        gesture_text,
        (22, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if debug_text:
        cv2.putText(
            frame,
            debug_text,
            (22, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    fps_overlay = frame.copy()
    cv2.rectangle(fps_overlay, (frame.shape[1] - 152, 10), (frame.shape[1] - 10, 54), (18, 18, 18), -1)
    cv2.addWeighted(fps_overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (frame.shape[1] - 152, 10), (frame.shape[1] - 10, 54), (120, 120, 120), 1)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (frame.shape[1] - 136, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def log_debug(message: str) -> None:
    """Print a timestamped debug message."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def get_label(handedness) -> str:
    """Safely read the MediaPipe handedness label."""
    if not handedness:
        return "Right"

    if isinstance(handedness, (list, tuple)) and handedness:
        category = handedness[0]
        for attribute in ("category_name", "display_name", "label"):
            value = getattr(category, attribute, None)
            if value:
                return value

    if hasattr(handedness, "classification") and handedness.classification:
        category = handedness.classification[0]
        for attribute in ("label", "display_name", "category_name"):
            value = getattr(category, attribute, None)
            if value:
                return value

    return "Right"
