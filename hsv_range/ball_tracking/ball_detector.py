import cv2

class Ball_Detector:
    def __init__(self, max_balls: int, ball_color_lower: tuple, ball_color_upper: tuple):
        self.__max_balls = max_balls
        self.__ball_color_lower = ball_color_lower
        self.__ball_color_upper = ball_color_upper

    def remove_other_colors(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.__ball_color_lower, self.__ball_color_upper)
        # opening operation to remove noise
        color_mask = cv2.erode(color_mask, None, iterations=2)
        color_mask = cv2.dilate(color_mask, None, iterations=2)
        return color_mask
    
    def get_max_contours(self, mask: cv2.typing.MatLike):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 50]
    
        if len(contours) <= 0:
            return None
        
        # finds top max_balls contours
        max_balls_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.__max_balls]
        
        return max_balls_contours