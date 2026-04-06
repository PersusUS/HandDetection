Proyecto: HandDetection

Descripción
- Control del cursor del ratón por gestos de mano usando MediaPipe Hand Landmarker.

Requisitos
- Python 3.8 o superior
- Cámara web
- Dependencias Python: `opencv-python`, `mediapipe`, `pynput`, `numpy`

Instalación
1. Crea y activa un entorno virtual (opcional pero recomendado).
2. Instala dependencias:

```
pip install opencv-python mediapipe pynput numpy
```

Archivos importantes
- `main.py`: punto de entrada. Contiene la lógica de detección y control del ratón.
- `hand_landmarker.task`: modelo usado por MediaPipe (debe estar en el mismo directorio).

Uso
1. Conecta una cámara y verifica que funcione.
2. Ejecuta:

```
python main.py
```

Notas
- Ajusta las constantes al inicio de `main.py` para sensibilidad, cámara y debug.
- Si usas Windows, es posible que necesites ejecutar el script con permisos adecuados para controlar el ratón.

Licencia
- Sin licencia especificada. Añade un archivo `LICENSE` si quieres publicar con una licencia concreta.
