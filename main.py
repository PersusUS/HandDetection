import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from pynput.mouse import Controller, Button
import math

# --- CONFIGURACIÓN PRINCIPAL ---
CAMERA_ID = 0
MOUSE_SENSITIVITY = 2.5
PINCH_THRESHOLD_RATIO = 0.15 # Proporción respecto a la caja delimitadora de la mano
CLICK_COOLDOWN_MS = 300
DEBUG = True

# Parámetros del 1 Euro Filter
FILTER_FREQ = 60.0
FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 0.5
FILTER_D_CUTOFF = 1.0

# --- CLASE: 1 EURO FILTER ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        t_e = t - self.t_prev if self.t_prev is not None else 1.0/FILTER_FREQ
        if t_e <= 0.0: t_e = 1.0/FILTER_FREQ
        
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

# --- CLASE: HAND TRACKER ---
class HandTracker:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, img, timestamp_ms):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def draw(self, img, results):
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for mark in hand_landmarks:
                    cv2.circle(img, (int(mark.x * img.shape[1]), int(mark.y * img.shape[0])), 3, (0, 255, 0), cv2.FILLED)
        return img

# --- CLASE: MOUSE CONTROLLER ---
class MouseController:
    def __init__(self):
        self.mouse = Controller()
        t = time.time()
        self.filter_x = OneEuroFilter(t, 0.0, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=FILTER_D_CUTOFF)
        self.filter_y = OneEuroFilter(t, 0.0, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=FILTER_D_CUTOFF)
        self.prev_pinch_pos = None
        self.prev_left_click_active = False
        self.prev_right_click_active = False

    def update_state(self, pinch_active, mid_x, mid_y, left_click_active, right_click_active):
        # 1. Clics (Edge Triggering)
        if left_click_active and not self.prev_left_click_active:
            self.mouse.click(Button.left)
            if DEBUG: print("L-CLICK")
        elif right_click_active and not self.prev_right_click_active:
            self.mouse.click(Button.right)
            if DEBUG: print("R-CLICK")

        self.prev_left_click_active = left_click_active
        self.prev_right_click_active = right_click_active

        # 2. Movimiento Relativo
        t = time.time()
        if not pinch_active:
            self.prev_pinch_pos = None
            return

        if self.prev_pinch_pos is not None:
            dx = (mid_x - self.prev_pinch_pos[0]) * MOUSE_SENSITIVITY
            dy = (mid_y - self.prev_pinch_pos[1]) * MOUSE_SENSITIVITY

            fdx = self.filter_x(t, dx)
            fdy = self.filter_y(t, dy)

            self.mouse.move(fdx, fdy)

        self.prev_pinch_pos = (mid_x, mid_y)
        if self.prev_pinch_pos is None:
             self.filter_x = OneEuroFilter(t, 0.0, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=FILTER_D_CUTOFF)
             self.filter_y = OneEuroFilter(t, 0.0, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA, d_cutoff=FILTER_D_CUTOFF)

# --- MAIN ---
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    tracker = HandTracker()
    mouse_ctrl = MouseController()

    print("--- HCI MOUSE CONTROL ---")

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        h_img, w_img, _ = img.shape
        
        timestamp_ms = int(time.time() * 1000)
        results = tracker.process(img, timestamp_ms)

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                label = handedness[0].category_name
                
                lm_list = hand_landmarks
                
                if label == "Left":  # Invertido por flip: La mano física derecha controla Movimiento y Clics unidos
                    x_coords = [lm.x for lm in lm_list]
                    y_coords = [lm.y for lm in lm_list]
                    
                    bbox_width = max(x_coords) - min(x_coords)
                    bbox_height = max(y_coords) - min(y_coords)
                    bbox_size = (bbox_width + bbox_height) / 2.0
                    
                    wrist_x, wrist_y = lm_list[0].x, lm_list[0].y
                    thumb_x, thumb_y = lm_list[4].x, lm_list[4].y
                    index_x, index_y = lm_list[8].x, lm_list[8].y
                    middle_x, middle_y = lm_list[12].x, lm_list[12].y
                    pinky_x, pinky_y = lm_list[20].x, lm_list[20].y
                    
                    # Pellizco Principal (Movimiento) -> Distancia Índice-Pulgar
                    dist_pinch = math.hypot(index_x - thumb_x, index_y - thumb_y)
                    pinch_active = dist_pinch < (bbox_size * PINCH_THRESHOLD_RATIO)

                    # Clic Derecho -> Cerrar la mano (distancia del Medio y Meñique a la muñeca)
                    dist_middle_wrist = math.hypot(middle_x - wrist_x, middle_y - wrist_y)
                    dist_pinky_wrist = math.hypot(pinky_x - wrist_x, pinky_y - wrist_y)
                    right_click_active = (dist_middle_wrist < bbox_size * 0.55) and (dist_pinky_wrist < bbox_size * 0.55)

                    # Clic Izquierdo -> Acercar Medio (Corazón) al Pulgar (excluyendo cuando se cierra toda la mano)
                    dist_left = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
                    left_click_active = (dist_left < (bbox_size * PINCH_THRESHOLD_RATIO * 1.5)) and not right_click_active
                    
                    mid_x = (thumb_x + index_x) / 2.0 * w_img
                    mid_y = (thumb_y + index_y) / 2.0 * h_img

                    mouse_ctrl.update_state(pinch_active, mid_x, mid_y, left_click_active, right_click_active)
                
                # La mano 'Right' (Izquierda física) queda libre de operaciones

        if DEBUG:
            img = tracker.draw(img, results)
            cv2.imshow("HCI", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
