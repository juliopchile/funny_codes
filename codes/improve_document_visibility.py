# requeriments openvc numpy pil

import cv2
import numpy as np
from PIL import Image

def update(_=None):
    """Callback de los trackbars: recalcula y muestra el umbralizado."""
    method = cv2.getTrackbarPos('Método', 'Thresh')
    thresh_val = cv2.getTrackbarPos('Umbral', 'Thresh')
    block_size = cv2.getTrackbarPos('BlockSize', 'Thresh') * 2 + 3  # impar ≥3
    C = cv2.getTrackbarPos('C', 'Thresh')

    if method == 0:
        # Umbral binario simple
        _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    else:
        # Umbral adaptativo gaussiano
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )
    cv2.imshow('Thresh', th)
    return th


if __name__ == '__main__':
    # 1) Carga y preprocesado inicial
    img = cv2.imread('cropped_output.png')
    if img is None:
        print("Error al leer la imagen.")
        exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=30)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    gray = clahe.apply(gray)

    # 2) Ventanas
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.imshow('Original', gray)
    cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)

    # 3) Trackbars
    # Método: 0 = binario, 1 = adaptativo
    cv2.createTrackbar('Método', 'Thresh', 0, 1, update)
    cv2.createTrackbar('Umbral', 'Thresh', 128, 255, update)
    cv2.createTrackbar('BlockSize', 'Thresh', 4, 25, update)  # se multiplica en update
    cv2.createTrackbar('C', 'Thresh', 2, 20, update)

    # 4) Primera llamada para mostrar algo
    th_img = update()

    print("Ajusta sliders. Pulsa 'p' para guardar como PDF, 'q' para salir.")

    # 5) Bucle principal
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            # Guardar en PDF
            pil = Image.fromarray(th_img).convert('L')
            pil.save('document_enhanced.pdf', 'PDF', resolution=100.0)
            print("Guardado: document_enhanced.pdf")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
