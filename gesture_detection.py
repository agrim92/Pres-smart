"""Gesture classification and motion-based swipe detection."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from utils import count_extended_fingers, euclidean_distance, get_finger_states, get_landmarks


@dataclass
class MotionSample:
    """A single fingertip motion sample."""

    timestamp: float
    x: float
    y: float


@dataclass
class GestureResult:
    """Structured output for the current frame."""

    gesture_name: str
    action: Optional[str]
    confidence: float
    finger_count: int
    raw_gesture: str = "unknown"
    raw_confidence: float = 0.0
    motion_velocity: float = 0.0


class GestureDetector:
    """Detect gestures and motion-driven swipe actions."""

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        action_cooldown: float = 1.2,
        motion_history_size: int = 15,
        motion_window_seconds: float = 0.75,
        swipe_distance_threshold: float = 0.14,
        min_swipe_velocity: float = 0.45,
        max_vertical_ratio: float = 0.55,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.action_cooldown = action_cooldown
        self.motion_history_size = motion_history_size
        self.motion_window_seconds = motion_window_seconds
        self.swipe_distance_threshold = swipe_distance_threshold
        self.min_swipe_velocity = min_swipe_velocity
        self.max_vertical_ratio = max_vertical_ratio

        self.motion_history: Deque[MotionSample] = deque(maxlen=motion_history_size)
        self.last_action_time = 0.0
        self.last_two_finger_time = 0.0

    def analyze(self, hand_landmarks, handedness_label: str) -> GestureResult:
        """Return a temporally smoothed gesture and any swipe action."""
        now = time.time()
        finger_states = get_finger_states(hand_landmarks, handedness_label)
        finger_count = count_extended_fingers(finger_states)

        gesture_name, confidence = self._classify_pose(hand_landmarks, finger_states, finger_count)
        raw_gesture = gesture_name
        raw_confidence = confidence

        if raw_gesture == "two_fingers":
            self.last_two_finger_time = now
            self._update_motion_history(now, hand_landmarks)
        elif now - self.last_two_finger_time > self.motion_window_seconds:
            self.motion_history.clear()

        action = None
        motion_velocity = self._estimate_velocity()
        if gesture_name == "two_fingers" and confidence >= self.confidence_threshold:
            action = self._detect_swipe(now)
            motion_velocity = self._estimate_velocity()

        return GestureResult(
            gesture_name=gesture_name,
            action=action,
            confidence=confidence,
            finger_count=finger_count,
            raw_gesture=raw_gesture,
            raw_confidence=raw_confidence,
            motion_velocity=motion_velocity,
        )

    def reset_swipe_tracking(self) -> None:
        """Clear motion state for this hand."""
        self.motion_history.clear()
        self.last_two_finger_time = 0.0

    def _classify_pose(self, hand_landmarks, finger_states, finger_count: int) -> tuple[str, float]:
        landmarks = get_landmarks(hand_landmarks)
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_size = max(euclidean_distance(wrist, middle_mcp), 0.01)

        tip_distances = {
            "thumb": euclidean_distance(landmarks[4], wrist) / palm_size,
            "index": euclidean_distance(landmarks[8], wrist) / palm_size,
            "middle": euclidean_distance(landmarks[12], wrist) / palm_size,
            "ring": euclidean_distance(landmarks[16], wrist) / palm_size,
            "pinky": euclidean_distance(landmarks[20], wrist) / palm_size,
        }
        average_non_thumb_distance = (
            tip_distances["index"]
            + tip_distances["middle"]
            + tip_distances["ring"]
            + tip_distances["pinky"]
        ) / 4.0

        if (
            finger_count <= 1
            and average_non_thumb_distance < 1.55
            and tip_distances["thumb"] < 1.85
        ):
            confidence = min(1.0, (1.55 - average_non_thumb_distance) / 0.45 + 0.55)
            return "fist", confidence

        if finger_count >= 4 and average_non_thumb_distance > 1.85:
            confidence = min(1.0, (average_non_thumb_distance - 1.85) / 0.35 + 0.65)
            return "open_hand", confidence

        if (
            finger_states["index"]
            and finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
            and finger_count in (2, 3)
        ):
            confidence = 0.88 if finger_count == 2 else 0.8
            return "two_fingers", confidence

        return "unknown", 0.0

    def _update_motion_history(self, now: float, hand_landmarks) -> None:
        landmarks = get_landmarks(hand_landmarks)
        index_tip = landmarks[8]
        self.motion_history.append(MotionSample(now, index_tip.x, index_tip.y))

        while self.motion_history and now - self.motion_history[0].timestamp > self.motion_window_seconds:
            self.motion_history.popleft()

    def _estimate_velocity(self) -> float:
        if len(self.motion_history) < 2:
            return 0.0

        first = self.motion_history[0]
        last = self.motion_history[-1]
        duration = last.timestamp - first.timestamp
        if duration <= 0:
            return 0.0

        distance = math.hypot(last.x - first.x, last.y - first.y)
        return distance / duration

    def _detect_swipe(self, now: float) -> Optional[str]:
        if now - self.last_action_time < self.action_cooldown:
            return None

        if len(self.motion_history) < max(6, self.motion_history_size // 2):
            return None

        points = list(self.motion_history)
        first = points[0]
        last = points[-1]
        total_dx = last.x - first.x
        total_dy = last.y - first.y
        total_distance = math.hypot(total_dx, total_dy)
        duration = last.timestamp - first.timestamp

        if duration <= 0 or total_distance < self.swipe_distance_threshold:
            return None

        if abs(total_dx) < self.swipe_distance_threshold:
            return None

        if abs(total_dy) > abs(total_dx) * self.max_vertical_ratio:
            return None

        velocity = total_distance / duration
        if velocity < self.min_swipe_velocity:
            return None

        significant_steps = []
        for previous, current in zip(points, points[1:]):
            step_dx = current.x - previous.x
            step_dy = current.y - previous.y
            step_distance = math.hypot(step_dx, step_dy)
            if step_distance > 0.002:
                significant_steps.append((step_dx, step_dy))

        if len(significant_steps) < 4:
            return None

        horizontal_sign = 1 if total_dx > 0 else -1
        consistent_steps = [
            step
            for step in significant_steps
            if step[0] * horizontal_sign > 0 and abs(step[0]) >= abs(step[1])
        ]
        direction_consistency = len(consistent_steps) / len(significant_steps)
        if direction_consistency < 0.75:
            return None

        self.last_action_time = now
        self.motion_history.clear()
        return "swipe_right" if total_dx > 0 else "swipe_left"
