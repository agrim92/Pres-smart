"""Webcam-driven presentation controller using hand gestures."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui

from gesture_detection import GestureDetector, GestureResult
from utils import draw_hand_landmarks, draw_overlay, get_label, get_landmarks, log_debug


@dataclass
class HandAnalysis:
    """Gesture analysis for a single detected hand."""

    slot_id: str
    label: str
    result: GestureResult


def handle_slide_action(action: str) -> str:
    """Translate a swipe action into a keyboard press."""
    if action == "swipe_right":
        pyautogui.keyDown("right")
        time.sleep(0.06)
        pyautogui.keyUp("right")
        log_debug("Swipe right detected -> next slide")
        return "Swipe Right -> Next Slide"

    if action == "swipe_left":
        pyautogui.keyDown("left")
        time.sleep(0.06)
        pyautogui.keyUp("left")
        log_debug("Swipe left detected -> previous slide")
        return "Swipe Left -> Previous Slide"

    return "No action"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gesture-controlled presentation assistant")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="MediaPipe minimum detection confidence",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.6,
        help="MediaPipe minimum tracking confidence",
    )
    parser.add_argument(
        "--swipe-threshold",
        type=float,
        default=0.14,
        help="Normalized fingertip travel required to trigger a swipe",
    )
    parser.add_argument(
        "--action-cooldown",
        type=float,
        default=1.2,
        help="Seconds to wait before allowing another swipe action",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="hand_landmarker.task",
        help="Path to the MediaPipe hand landmarker model file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_path).resolve()

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0

    def make_detector() -> GestureDetector:
        return GestureDetector(
            swipe_distance_threshold=args.swipe_threshold,
            action_cooldown=args.action_cooldown,
        )

    detectors = {
        "slot_0": make_detector(),
        "slot_1": make_detector(),
    }
    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check that a camera is connected.")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}. "
            "Download hand_landmarker.task and pass it with --model-path if needed."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    interaction_mode = False
    gesture_text = "Waiting for hand..."
    debug_text = "Press Q to quit"
    fps = 0.0
    previous_frame_time = time.monotonic()

    log_debug("Application started")

    with mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                log_debug("Failed to read frame from webcam")
                break

            frame_time = time.monotonic()
            frame_delta = max(frame_time - previous_frame_time, 1e-6)
            instantaneous_fps = 1.0 / frame_delta
            fps = instantaneous_fps if fps == 0.0 else (0.85 * fps + 0.15 * instantaneous_fps)
            previous_frame_time = frame_time

            # Mirror the preview so it feels natural for the presenter.
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.monotonic() * 1000)
            results = hands.detect_for_video(mp_image, timestamp_ms)

            gesture_text = "No hand detected"
            debug_text = "Press Q to quit"

            if results.hand_landmarks:
                analyses = []
                indexed_hands = []
                for index, hand_landmarks in enumerate(results.hand_landmarks):
                    landmarks = get_landmarks(hand_landmarks)
                    average_x = sum(landmark.x for landmark in landmarks) / len(landmarks)
                    indexed_hands.append((average_x, index, hand_landmarks))

                indexed_hands.sort(key=lambda item: item[0])
                seen_slots = set()
                for slot_number, (_, index, hand_landmarks) in enumerate(indexed_hands):
                    slot_id = f"slot_{slot_number}"
                    handedness = results.handedness[index] if results.handedness and index < len(results.handedness) else None
                    handedness_label = get_label(handedness)
                    detector = detectors.setdefault(slot_id, make_detector())
                    result = detector.analyze(hand_landmarks, handedness_label)
                    analyses.append(HandAnalysis(slot_id=slot_id, label=handedness_label, result=result))
                    seen_slots.add(slot_id)
                    draw_hand_landmarks(frame, hand_landmarks)

                for slot_id, detector in detectors.items():
                    if slot_id not in seen_slots:
                        detector.reset_swipe_tracking()

                fist_hands = [
                    analysis
                    for analysis in analyses
                    if analysis.result.raw_gesture == "fist" and analysis.result.raw_confidence >= 0.7
                ]
                active_fist_slots = {analysis.slot_id for analysis in fist_hands}
                previous_interaction_mode = interaction_mode
                interaction_mode = bool(active_fist_slots)

                if interaction_mode and not previous_interaction_mode:
                    log_debug("Fist held -> interaction mode ON")
                elif previous_interaction_mode and not interaction_mode:
                    log_debug("Fist released -> interaction mode OFF")
                    for detector in detectors.values():
                        detector.reset_swipe_tracking()

                swipe_hands = [
                    analysis
                    for analysis in analyses
                    if analysis.slot_id not in active_fist_slots
                ]
                swipe_action = next(
                    (analysis.result.action for analysis in swipe_hands if analysis.result.action),
                    None,
                )
                ready_swipe_hand = next(
                    (
                        analysis
                        for analysis in swipe_hands
                        if analysis.result.gesture_name == "two_fingers"
                    ),
                    None,
                )

                if interaction_mode and swipe_action:
                    gesture_text = handle_slide_action(swipe_action)
                elif interaction_mode and ready_swipe_hand:
                    gesture_text = "Swipe hand ready"
                elif interaction_mode:
                    gesture_text = "Hold fist + swipe with other hand"
                elif any(analysis.result.raw_gesture == "two_fingers" for analysis in analyses):
                    gesture_text = "Need fist on other hand"
                else:
                    gesture_text = "Show fist to enable control"

                debug_text = " | ".join(
                    (
                        f"{analysis.label}/{analysis.slot_id}: "
                        f"{analysis.result.gesture_name or 'unknown'} "
                        f"{analysis.result.confidence:.2f} "
                        f"(raw {analysis.result.raw_gesture} {analysis.result.raw_confidence:.2f})"
                    )
                    for analysis in analyses
                )
            else:
                if interaction_mode:
                    log_debug("No hands detected -> interaction mode OFF")
                    for detector in detectors.values():
                        detector.reset_swipe_tracking()
                interaction_mode = False

            draw_overlay(frame, interaction_mode, gesture_text, debug_text, fps=fps)
            cv2.imshow("Gesture Presentation Assistant", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    log_debug("Application stopped")


if __name__ == "__main__":
    main()
