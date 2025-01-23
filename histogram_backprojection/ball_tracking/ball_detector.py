import cv2
import numpy as np

class Ball_Detector:
    def __init__(self, max_balls: int, reference_image: cv2.typing.MatLike, h_bins: int = 10, s_bins: int = 20):
        self.__max_balls = max_balls
        
        # Farbhistogramm aus einer Referenzmaske erstellen
        self.__hist = self.__create_histogram(reference_image, h_bins, s_bins)

    def __create_histogram(self, ref_image: cv2.typing.MatLike, h_bins: int, s_bins: int):
        """Erstellt ein Farbhistogramm basierend auf einem Referenzbild."""
        hsv_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_ref, (0, 50, 50), (180, 255, 255))  # Anpassbarer Bereich
        hist = cv2.calcHist([hsv_ref], [0, 1], mask, [h_bins, s_bins], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return hist

    def remove_other_colors(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Erzeugt eine Wahrscheinlichkeitskarte basierend auf dem Histogramm."""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv_frame], [0, 1], self.__hist, [0, 180, 0, 256], scale=1)
        
        # Glättung und Schwellenwertanwendung
        back_proj = cv2.GaussianBlur(back_proj, (11, 11), 0)
        _, mask = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)  # Schwellenwert anpassen
        return mask

    def get_max_contours(self, mask: cv2.typing.MatLike):
        """Findet die größten Konturen im Bild."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 50]
    
        if len(contours) <= 0:
            return None
        
        max_balls_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.__max_balls]
        return max_balls_contours
