import numpy as np

from hand_tracking.hand import Hand

class Hand_Tracker:
    def __init__(self, left_hand: Hand, right_hand: Hand, detection_threshold=float("inf")):
        self.__left_hand = left_hand
        self.__right_hand = right_hand
        self.__detection_treshold = detection_threshold

    def __is_first_detection(self):
        detection_history_left = self.__left_hand.get_coord_history()
        detection_history_right = self.__right_hand.get_coord_history()
        return len(detection_history_left) == 0 and len(detection_history_right) == 0
    
    def __determine_left_and_right(self, detections: list):
        x1, _ = detections[0]
        x2, _ = detections[1]

        if x1 < x2:
            return [detections[1], detections[0]]
        else:
            return [detections[0], detections[1]]
        
    def __calculate_distance(self, detection, hand: Hand):
        hand_x, _ = hand.get_coords()
        detection_x, _ = detection

        return abs(hand_x - detection_x)

    def update(self, detections):
        if len(detections) == 0:
            return
        
        if self.__is_first_detection():
            if len(detections) < 2:
                return
            
            left_coords, right_coords = self.__determine_left_and_right(detections)
            self.__left_hand.update_coords(left_coords)
            self.__right_hand.update_coords(right_coords)
            return
        
        if len(detections) == 2:
            left_coords, right_coords = self.__determine_left_and_right(detections)
            left_distance = self.__calculate_distance(left_coords, self.__left_hand)
            right_distance = self.__calculate_distance(right_coords, self.__right_hand)

            if left_distance < self.__detection_treshold:
                self.__left_hand.update_coords(left_coords)
            if right_distance < self.__detection_treshold:
                self.__right_hand.update_coords(right_coords)
            return
        
        if len(detections) == 1:
            detection = detections[0]
            left_distance = self.__calculate_distance(detection, self.__left_hand)
            right_distance = self.__calculate_distance(detection, self.__right_hand)
            
            if left_distance < right_distance:
                if left_distance < self.__detection_treshold:
                    self.__left_hand.update_coords(detection)
            else:
                if right_distance < self.__detection_treshold:
                    self.__right_hand.update_coords(detection)
            return