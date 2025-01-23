import numpy as np

class Hand:
    def __init__(self, id, bbox_area = 15000):
        self.__id = id
        self.__current_coords = []
        self.__coord_history = []
        self.__bbox = []
        self.__bbox_area = bbox_area

    def __update_bbox(self):
        x, y = self.get_coords()
        side_length = int(np.sqrt(self.__bbox_area))
        half_side = side_length // 2
        top_left = (x - half_side, y - half_side)
        bottom_right = (x + half_side, y + half_side + 400) # Extend the square downwards
        self.__bbox = [top_left, bottom_right]

    def update_coords(self, coords):
        self.__coord_history.append(self.__current_coords)
        self.__current_coords = coords
        self.__update_bbox()

    def get_coords(self):
        return self.__current_coords
    
    def get_coord_history(self):
        return self.__coord_history
    
    def get_id(self):
        return self.__id
    
    def get_bbox(self):
        return self.__bbox