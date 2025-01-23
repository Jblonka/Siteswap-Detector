import numpy as np

from hand_tracking.hand import Hand

class Catch_Detector:
    def __init__(self, max_balls: int, left_hand: Hand, right_hand: Hand):
        self.__left_hand: Hand = left_hand
        self.__right_hand: Hand = right_hand
        self.__balls_center_id = []
        self.__left_hand_state = {i: False for i in range(1, max_balls + 1)}
        self.__right_hand_state = {i: False for i in range(1, max_balls + 1)}
        self.__left_hand_exit_count = {i: 0 for i in range(1, max_balls + 1)}
        self.__right_hand_exit_count = {i: 0 for i in range(1, max_balls + 1)}
        self.__catch_history = []
        self.__exit_threshold = 5  # Number of frames the ball must be outside the hand to be registered as caught again

    def __ball_is_in_hand(self, ball_center_id, hand: Hand):
        hand_top_left, hand_bottom_right = hand.get_bbox()
        x, y, id = ball_center_id
        return hand_top_left[0] <= x <= hand_bottom_right[0] and hand_top_left[1] <= y <= hand_bottom_right[1]

    def __calculate_distance_to_hand(self, ball_center_id, hand: Hand):
        hand_x, hand_y = hand.get_coords()
        ball_x, ball_y, id = ball_center_id
        
        return np.linalg.norm(np.array((hand_x, hand_y)) - np.array((ball_x, ball_y)))

    def __determine_catch_state(self):
        # hands have not been initialized yet
        if len(self.__left_hand.get_coords()) == 0 or len(self.__right_hand.get_coords()) == 0:
            return
        
        for ball_center_id in self.__balls_center_id:
            ball_x, ball_y, ball_id = ball_center_id
            in_left_hand = self.__ball_is_in_hand(ball_center_id, self.__left_hand)
            in_right_hand = self.__ball_is_in_hand(ball_center_id, self.__right_hand)

            # bboxes overlap
            if in_left_hand and in_right_hand:
                distance_to_left = self.__calculate_distance_to_hand(ball_center_id, self.__left_hand)
                distance_to_right = self.__calculate_distance_to_hand(ball_center_id, self.__right_hand)
                if distance_to_left < distance_to_right:
                    if not self.__left_hand_state[ball_id]:
                        self.__left_hand_state[ball_id] = True
                        self.__right_hand_state[ball_id] = False
                        self.__catch_history.append((ball_id, 'left'))
                else:
                    if not self.__right_hand_state[ball_id]:
                        self.__right_hand_state[ball_id] = True
                        self.__left_hand_state[ball_id] = False
                        self.__catch_history.append((ball_id, 'right'))
                return
            
            if in_left_hand:
                self.__left_hand_exit_count[ball_id] = 0
                if not self.__left_hand_state[ball_id]:
                    self.__left_hand_state[ball_id] = True
                    self.__right_hand_state[ball_id] = False
                    self.__catch_history.append((ball_id, 'left'))
            elif in_right_hand:
                self.__right_hand_exit_count[ball_id] = 0
                if not self.__right_hand_state[ball_id]:
                    self.__right_hand_state[ball_id] = True
                    self.__left_hand_state[ball_id] = False
                    self.__catch_history.append((ball_id, 'right'))
            else:
                if self.__left_hand_state[ball_id]:
                    self.__left_hand_exit_count[ball_id] += 1
                    if self.__left_hand_exit_count[ball_id] >= self.__exit_threshold:
                        self.__left_hand_state[ball_id] = False
                if self.__right_hand_state[ball_id]:
                    self.__right_hand_exit_count[ball_id] += 1
                    if self.__right_hand_exit_count[ball_id] >= self.__exit_threshold:
                        self.__right_hand_state[ball_id] = False

    def update(self, balls_center_id):
        self.__balls_center_id = balls_center_id
        self.__determine_catch_state()

    def get_catch_history(self):
        return self.__catch_history

    def get_simple_catch_history(self):
        return [ball_id for ball_id, _ in self.__catch_history]
