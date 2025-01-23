import math
import numpy

class Ball_Tracker():
    def __init__(self, max_balls):
        self.__max_count = max_balls
        self.__given_ids = []
        self.__prev_detections = []

    def __get_next_id(self):
        for i in range(1, self.__max_count + 1):
            if i not in self.__given_ids:
                self.__given_ids.append(i)
                return i
        return -1

    def __calculate_distance(self, detection, prev_detection):
        x1, y1 = detection
        x2, y2 = prev_detection
        return math.hypot(x1 - x2, y1 - y2)
    
    def __calculate_distance_matrix(self, detections, prev_detections):
        distance_matrix = numpy.zeros((len(detections), len(prev_detections)))
        for i, detection in enumerate(detections):
            for j, prev_detection in enumerate(prev_detections):
                distance_matrix[i][j] = self.__calculate_distance(detection, prev_detection[:2])
        return distance_matrix
    
    def update(self, detections):
        objects_center_id = []
        
        # initial ids
        if not self.__prev_detections:
            for detection in detections:
                x, y = detection
                objects_center_id.append([x, y, self.__get_next_id()])
            self.__prev_detections = objects_center_id
            return objects_center_id
        
        distance_matrix = self.__calculate_distance_matrix(detections, self.__prev_detections)

        used_detections = set()
        used_prev_detections = set()
        used_ids = []
        for _ in range(len(detections)):
            min_element = numpy.min(distance_matrix)
            if min_element == float('inf'):
                break
            min_index = numpy.unravel_index(numpy.argmin(distance_matrix), distance_matrix.shape)
            x, y = detections[min_index[0]]
            _, _, id_prev = self.__prev_detections[min_index[1]]
            objects_center_id.append([x, y, id_prev])
            used_ids.append(id_prev)
            distance_matrix[min_index[0], :] = float('inf')  # Mark this row as used
            distance_matrix[:, min_index[1]] = float('inf')  # Mark this column as used
            used_detections.add(min_index[0])
            used_prev_detections.add(min_index[1])

        # Handle unassigned detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                objects_center_id.append([detection[0], detection[1], self.__get_next_id()])

        # Handle unassigned prev_detections
        for i, prev_detection in enumerate(self.__prev_detections):
            if i not in used_prev_detections:
                objects_center_id.append([prev_detection[0], prev_detection[1], prev_detection[2]])

        self.__prev_detections = objects_center_id
        return objects_center_id

# if __name__ == '__main__':
#     tracker = EuclideanDistTracker(3)
#     detections = [[2.12, 2.23], [2.10, 2.23], [1.23, 0.23]]
#     prev_detections = [[1.8, 1.564, 1], [1.23, 1, 2], [3.23, 3.43, 3]]
#     matrix = tracker.calculate_distance_matrix(detections, prev_detections)
#     # print('Distance matrix:', matrix)
#     min_element = numpy.min(matrix)
#     min_index = numpy.unravel_index(numpy.argmin(matrix), matrix.shape)

#     # print('Smallest element in the matrix:', min_element)
#     # print('Index of the smallest element:', min_index)
#     for detection in detections:
#         closest_prev_detection, min_distance = tracker.find_closest_prev_detection(detection, prev_detections)
#         print('Detection:', detection, '|', 'Closest prev_detection:', closest_prev_detection, '| Distance:', min_distance)
#     iniial_objects = tracker.update(detections, [])
#     print('Initial objects:', iniial_objects)
#     objects_center_id = tracker.update(detections, prev_detections)
#     print('Objects center id:', objects_center_id)
