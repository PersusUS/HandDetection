import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0  # Posición anterior para suavizado
smoothing_factor = 0.2  # Factor para suavizado del movimiento del ratón
sensitivity = 1.5  # Sensibilidad para ajustar la velocidad del ratón

def is_finger_bent(finger_tip, finger_dip):
    """ Verifica si un dedo está doblado. """
    return finger_tip[1] > finger_dip[1]

def move_mouse_smooth(x, y):
    """ Suaviza el movimiento del ratón para evitar saltos bruscos. """
    global prev_x, prev_y
    x = int(prev_x + (x - prev_x) * smoothing_factor)
    y = int(prev_y + (y - prev_y) * smoothing_factor)
    prev_x, prev_y = x, y
    pyautogui.moveTo(x, y)

def click_mouse(button):
    """ Simula un clic de ratón con un botón especificado. """
    pyautogui.click(button=button)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            index_tip = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            thumb_tip = (landmarks[mp_hands.HandLandmark.THUMB_TIP].x * w, landmarks[mp_hands.HandLandmark.THUMB_TIP].y * h)
            middle_tip = (landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h)
            middle_dip = (landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * w, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * h)
            ring_tip = (landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x * w, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * h)
            ring_dip = (landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x * w, landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y * h)

            # Verificar si el índice y el pulgar están cerca (gesto de pellizco)
            if abs(index_tip[0] - thumb_tip[0]) < 40 and abs(index_tip[1] - thumb_tip[1]) < 40:
                # Escalar las posiciones del índice a la pantalla y suavizar el movimiento
                screen_x = int(index_tip[0] * screen_width / w * sensitivity)
                screen_y = int(index_tip[1] * screen_height / h * sensitivity)
                move_mouse_smooth(screen_x, screen_y)

                # Clic izquierdo si el dedo medio está doblado
                if is_finger_bent(middle_tip, middle_dip):
                    click_mouse('left')

                # Clic derecho si los dedos medio y anular están doblados
                if is_finger_bent(middle_tip, middle_dip) and is_finger_bent(ring_tip, ring_dip):
                    click_mouse('right')

    cv2.imshow('Hand Gesture Mouse Control', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
