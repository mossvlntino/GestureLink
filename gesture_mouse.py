"""
Gesture Mouse — Left Hand = Mode (1..3), Right Hand = Action
------------------------------------------------------------
Modes (hold LEFT hand fingers):
  1 → Mode 1: Cursor + Clicks
  2 → Mode 2: Brightness (right hand 1..5 fingers = 20..100%)
  3 → Mode 3: Volume    (right hand 1..5 fingers = 20..100%)
  0 or 5 → idle

Mode 1 (Cursor + Clicks) on RIGHT hand:
  • Move pointer with index fingertip
  • Index + Thumb pinch (short) → Left click
  • Index + Thumb hold → (ignored; no drag)
  • Middle + Thumb hold → Click-and-drag (hold = drag, release = drop)
  • Ring + Thumb tap → Right click

Hotkeys: [ / ] smoothing,  Q / ESC quit
Python 3.12, Windows:
  pip install mediapipe opencv-python numpy pyautogui pillow pycaw comtypes screen-brightness-control
"""

import sys, subprocess


def ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


for p in [
    "opencv-python",
    "mediapipe",
    "numpy",
    "pyautogui",
    "pillow",
    "pycaw",
    "comtypes",
    "screen-brightness-control",
]:
    ensure(p)

import time
import numpy as np
import cv2
import mediapipe as mp
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# ---- Version guard ----
if sys.version_info >= (3, 13):
    raise SystemExit("Run with Python 3.12.x — MediaPipe has no wheels for 3.13/3.14.")

# ---- Safety ----
pyautogui.FAILSAFE = True  # slam cursor to top-left to abort

# ---- System Volume (pycaw) ----
_speakers = AudioUtilities.GetSpeakers()
_interface = _speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
_volume = cast(_interface, POINTER(IAudioEndpointVolume))

# ---- Landmark indices ----
IDX_TIP, MID_TIP, RING_TIP, THUMB_TIP = 8, 12, 16, 4


def dist2d(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def landmark_px(lm, w, h, idx):
    p = lm[idx]
    return (p.x * w, p.y * h)


class Ema:
    def __init__(self, alpha=0.30):
        self.alpha, self.v = alpha, None

    def step(self, x, y):
        if self.v is None:
            self.v = (x, y)
        else:
            self.v = (
                (1 - self.alpha) * x + self.alpha * self.v[0],
                (1 - self.alpha) * y + self.alpha * self.v[1],
            )
        return self.v


def median3(q):
    if not q:
        return None
    xs = [p[0] for p in q[-3:]]
    ys = [p[1] for p in q[-3:]]
    xs.sort()
    ys.sort()
    return (xs[len(xs) // 2], ys[len(ys) // 2])


def step_towards(cur, tgt, max_step):
    if tgt > cur:
        return min(tgt, cur + max_step)
    if tgt < cur:
        return max(tgt, cur - max_step)
    return cur


def count_extended_fingers(lms, label, w, h):
    """Count 0..5 fingers. 4 fingers: tip.y < pip.y; thumb uses horizontal relation."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    ext = 0
    for t, p in zip(tips, pips):
        if (lms[t].y * h) < (lms[p].y * h):
            ext += 1
    tipx, ipx = lms[4].x * w, lms[3].x * w
    if str(label).lower().startswith("right"):
        if tipx < ipx - 5:
            ext += 1
    else:
        if tipx > ipx + 5:
            ext += 1
    return int(np.clip(ext, 0, 5))


def main():
    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera found.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame0 = cap.read()
    if not ok:
        print("❌ Can't read from camera.")
        return
    img0 = cv2.flip(frame0, 1)
    H, W = img0.shape[:2]

    # Screen
    screen_w, screen_h = pyautogui.size()
    print(f"Screen: {screen_w}x{screen_h}")

    # MediaPipe Hands
    mp_h = mp.solutions.hands
    mp_s = mp.solutions.drawing_styles
    hands = mp_h.Hands(
        max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6
    )

    # State
    smoothing = 0.30
    ema = Ema(alpha=smoothing)
    pos_hist = []
    DEADZONE_PX = 8
    DISPLAY_SCALE = 1.35
    zoom_out = 1.00

    # Drag & clicks
    dragging = False
    # timers for detecting short vs hold on index+thumb (left click only)
    idx_thumb_pinching = False
    idx_thumb_start = 0.0
    # timers for middle+thumb HOLD drag
    mid_thumb_pinching = False
    mid_thumb_start = 0.0
    # cooldown for ring+thumb right click
    RIGHT_COOLDOWN = 0.25
    last_right_time = 0.0

    # Brightness/Volume glide (targets updated only in modes 2/3)
    APPLY_EVERY, last_apply = 0.15, 0.0
    VOL_MAX_STEP = 0.03
    BRI_MAX_STEP = 4
    vol_target_scalar = None  # 0..1
    bri_target = None  # 10..100

    # UI
    hud_msg = (
        "LEFT selects mode (hold): 1 Cursor | 2 Brightness | 3 Volume. RIGHT performs."
    )
    mode_name = {1: "Cursor/Clicks", 2: "Brightness", 3: "Volume"}
    active_mode = None
    prev_pos = None

    cv2.namedWindow("Gesture Mouse (Left=Mode, Right=Action)", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.flip(frame, 1)

        # widest FOV (no crop)
        h, w = img.shape[:2]
        if zoom_out > 1.0:
            nw, nh = int(w / zoom_out), int(h / zoom_out)
            x1, y1 = (w - nw) // 2, (h - nh) // 2
            img = cv2.resize(img[y1 : y1 + nh, x1 : x1 + nw], (w, h))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # HUD
        cv2.rectangle(img, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.putText(
            img,
            "Gesture Mouse — Mode by LEFT, Action by RIGHT",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Smoothing: {smoothing:.2f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            img, hud_msg, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 200, 200), 2
        )

        left_count = right_count = None
        left_lm = right_lm = None
        now = time.time()

        if res.multi_hand_landmarks and res.multi_handedness:
            # Separate LEFT/RIGHT
            for lm, handness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handness.classification[0].label
                if label == "Left":
                    left_lm = (lm, label)
                elif label == "Right":
                    right_lm = (lm, label)

            # Draw both
            for pair in [left_lm, right_lm]:
                if pair:
                    lm, _ = pair
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        lm,
                        mp_h.HAND_CONNECTIONS,
                        mp_s.get_default_hand_landmarks_style(),
                        mp_s.get_default_hand_connections_style(),
                    )

            # ---- Active mode from LEFT hand (hold 1..3 fingers) ----
            if left_lm:
                lm, label = left_lm
                left_count = count_extended_fingers(lm.landmark, label, w, h)
                if 1 <= left_count <= 3:
                    active_mode = left_count
                else:
                    active_mode = None
            else:
                active_mode = None

            # ---- RIGHT hand performs according to mode ----
            if right_lm and active_mode is not None:
                lm, label = right_lm
                # points
                idx = landmark_px(lm.landmark, w, h, IDX_TIP)
                mid = landmark_px(lm.landmark, w, h, MID_TIP)
                ring = landmark_px(lm.landmark, w, h, RING_TIP)
                thu = landmark_px(lm.landmark, w, h, THUMB_TIP)

                if active_mode == 1:
                    # ----- Pointer movement (index fingertip) -----
                    tx = np.clip(idx[0] / w * screen_w, 0, screen_w - 1)
                    ty = np.clip(idx[1] / h * screen_h, 0, screen_h - 1)
                    pos_hist.append((tx, ty))
                    med = median3(pos_hist) or (tx, ty)
                    sx, sy = ema.step(med[0], med[1])
                    if prev_pos is None:
                        move = True
                    else:
                        dx = sx - prev_pos[0]
                        dy = sy - prev_pos[1]
                        move = (dx * dx + dy * dy) ** 0.5 > DEADZONE_PX
                    if move:
                        prev_pos = (sx, sy)
                        try:
                            pyautogui.moveTo(int(sx), int(sy), _pause=False)
                        except pyautogui.FailSafeException:
                            hud_msg = "Failsafe triggered. Move away from top-left."

                    # Distances (normalized)
                    idx_thumb = dist2d(idx, thu) / max(w, h)
                    mid_thumb = dist2d(mid, thu) / max(w, h)
                    ring_thumb = dist2d(ring, thu) / max(w, h)

                    PINCH_THRESH = 0.045

                    # --- Index+Thumb: LEFT CLICK only (no drag on hold) ---
                    if idx_thumb < PINCH_THRESH:
                        if not idx_thumb_pinching:
                            idx_thumb_pinching = True
                            idx_thumb_start = now
                        # (even if held, we ignore drag here)
                    else:
                        if idx_thumb_pinching:
                            pinch_len = now - idx_thumb_start
                            # treat as click regardless of hold length per your rule
                            pyautogui.click()
                            hud_msg = "Left click."
                            idx_thumb_pinching = False

                    # --- Middle+Thumb HOLD: DRAG ---
                    if mid_thumb < PINCH_THRESH:
                        if not mid_thumb_pinching:
                            mid_thumb_pinching = True
                            mid_thumb_start = now
                        else:
                            if not dragging and (now - mid_thumb_start) >= 0.35:
                                pyautogui.mouseDown()
                                dragging = True
                                hud_msg = "Dragging… (release middle+thumb to drop)"
                    else:
                        if mid_thumb_pinching:
                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False
                                hud_msg = "Drag ended."
                            mid_thumb_pinching = False

                    # --- Ring+Thumb TAP: RIGHT CLICK (with cooldown) ---
                    if (
                        ring_thumb < PINCH_THRESH
                        and (now - last_right_time) > RIGHT_COOLDOWN
                    ):
                        pyautogui.click(button="right")
                        hud_msg = "Right click."
                        last_right_time = now

                elif active_mode == 2:
                    # ----- Brightness (right-hand fingers 1..5 -> 20..100) -----
                    right_count = count_extended_fingers(lm.landmark, label, w, h)
                    if right_count and right_count >= 1:
                        bri_target = right_count * 20  # smooth glide applied below

                elif active_mode == 3:
                    # ----- Volume (right-hand fingers 1..5 -> 20..100) -----
                    right_count = count_extended_fingers(lm.landmark, label, w, h)
                    if right_count and right_count >= 1:
                        vol_target_scalar = (right_count * 20) / 100.0  # 0.20..1.0

        # ---- Apply brightness/volume glides regularly ----
        if time.time() - last_apply >= APPLY_EVERY:
            # Volume
            if vol_target_scalar is not None:
                try:
                    cur_scalar = float(_volume.GetMasterVolumeLevelScalar())
                except Exception:
                    cur_scalar = 0.0
                new_scalar = step_towards(cur_scalar, vol_target_scalar, VOL_MAX_STEP)
                _volume.SetMasterVolumeLevelScalar(new_scalar, None)
                cv2.putText(
                    img,
                    f"VOL {int(new_scalar*100)}%",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            # Brightness
            if bri_target is not None:
                try:
                    cur_list = sbc.get_brightness()
                    cur_bri = float(cur_list[0]) if cur_list else 50.0
                except Exception:
                    cur_bri = 50.0
                new_bri = step_towards(cur_bri, float(bri_target), BRI_MAX_STEP)
                try:
                    sbc.set_brightness(int(new_bri))
                except Exception:
                    pass
                cv2.putText(
                    img,
                    f"BRI {int(new_bri)}%",
                    (180, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
            last_apply = time.time()

        # Mode/Counts HUD line
        mode_text = f"Mode: {mode_name.get(active_mode,'(none)')}"
        if left_count is not None:
            mode_text += f"  |  Left fingers: {left_count}"
        if right_count is not None:
            mode_text += f"  |  Right fingers: {right_count}"
        cv2.putText(
            img, mode_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Display (bigger preview)
        disp_w, disp_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
        display_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Gesture Mouse (Left=Mode, Right=Action)", display_img)

        # Hotkeys
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("["):
            smoothing = float(np.clip(smoothing + 0.05, 0.05, 0.90))
            ema.alpha = smoothing
            hud_msg = f"Smoothing: {smoothing:.2f}"
        elif key == ord("]"):
            smoothing = float(np.clip(smoothing - 0.05, 0.05, 0.90))
            ema.alpha = smoothing
            hud_msg = f"Smoothing: {smoothing:.2f}"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
