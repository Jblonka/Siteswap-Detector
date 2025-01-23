from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from juggling_types import FramesArray, Frame

class Siteswap_Graph_Plotter():
    def __init__(self, max_balls: int):
        self.frames: FramesArray = []
        self.frames_by_ball = {i: [] for i in range(1, max_balls + 1)}
        self.hand_detections = []
        self.max_balls = max_balls
    
    def add_ball_detections(self, objects_center_id: Frame):
        # Save coordinates of balls in each frame
        if not objects_center_id:
            return
        
        self.frames.append(objects_center_id)
        
        for object in objects_center_id:
            x, y, ball_id = object
            self.frames_by_ball[ball_id].append((x, y))

    def add_hand_detections(self, objects_center_id: Frame):
        # Save coordinates of balls in each frame
        if not objects_center_id:
            return
        
        for object in objects_center_id:
            if object is not None:
                x, y, hand_id = object
                self.hand_detections.append((x, y))    
    
    def catch_history_patterns(self, catch_history, max_balls):
        def find_repeating_patterns(history):
            history = [tuple(item) if isinstance(item, list) else item for item in history]
            pattern_counts = defaultdict(int)
            n = len(history)
            balls_set = set(list(range(1, max_balls + 1)))
            
            # Check all possible sub-patterns
            for length in range(1, n // 2 + 1):
                start = 0
                while start <= n - length:
                    # Define the pattern and check consecutive occurrences
                    pattern = tuple(history[start:start + length])
                    occurrences = 1
                    while start + occurrences * length <= n - length and \
                        tuple(history[start + occurrences * length : start + (occurrences + 1) * length]) == pattern:
                        occurrences += 1
                    
                    # If repeated consecutively, count pattern and skip ahead
                    if occurrences > 1:
                        pattern_counts[pattern] += occurrences
                        start += occurrences * length
                    else:
                        start += 1

            # Filter out patterns that repeat consecutively at least once
            repeating_patterns = {pattern: count for pattern, count in pattern_counts.items() if count > 1}

            # Filter out patterns that don't contain all balls
            # Filter patterns
            valid_patterns = {
                pattern: count
                for pattern, count in repeating_patterns.items()
                if set(pattern) == balls_set
            }

            # Sort patterns by repetition times in descending order
            sorted_patterns = sorted(valid_patterns.items(), key=lambda item: item[1], reverse=True)
            
            return sorted_patterns

        repeating_patterns = find_repeating_patterns(catch_history)
        return repeating_patterns    
    
    def get_boundingbox(self):
        # Convert the detections to a NumPy array
        detections = np.array(self.hand_detections)

        # Use K-means clustering to separate the detections into two clusters
        kmeans = KMeans(n_clusters=2, random_state=0).fit(detections)
        labels = kmeans.labels_

        # Separate the detections into two groups based on the cluster labels
        group1 = detections[labels == 0]
        group2 = detections[labels == 1]

        # Calculate the bounding box for the first group
        if len(group1) > 0:
            min_x1, min_y1 = np.min(group1, axis=0)
            max_x1, max_y1 = np.max(group1, axis=0)
            bbox1 = [[min_x1, max_x1, max_x1, min_x1, min_x1], [min_y1, min_y1, max_y1, max_y1, min_y1]]
        else:
            bbox1 = None

        # Calculate the bounding box for the second group
        if len(group2) > 0:
            min_x2, min_y2 = np.min(group2, axis=0)
            max_x2, max_y2 = np.max(group2, axis=0)
            bbox2 = [[min_x2, max_x2, max_x2, min_x2, min_x2], [min_y2, min_y2, max_y2, max_y2, min_y2]]
        else:
            bbox2 = None

        return bbox1, bbox2

    def plot_pattern_graph(self):
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        plt.figure(figsize=(8, 8))
        for (ball_id, frames) in self.frames_by_ball.items():
            ballCoordinates = self.frames_by_ball.get(ball_id)
            ballX = [x for x, y in ballCoordinates]
            ballY = [y for x, y in ballCoordinates]

            plt.scatter(ballX, ballY, color=colors[ball_id % len(colors)], marker='o', label=f'Ball {ball_id}')
        
        bbox1, bbox2 = self.get_boundingbox()
        if bbox1 is not None and bbox2 is not None:
            plt.plot(bbox1[0], bbox1[1], color=colors[1 % len(colors)], linestyle='dashed', label=f'Bounding Box left hand')
            plt.plot(bbox2[0], bbox2[1], color=colors[2 % len(colors)], linestyle='dashed', label=f'Bounding Box right hand')
        # Labeling axes
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(f'2D Coordinate System with all detections')

        # Set grid and limits for visibility
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

        # Reverse the y-axis
        plt.gca().invert_yaxis()

        # Add a legend to differentiate the balls
        plt.legend()

        # Display the plot
        plt.show()

    def plot_hands_graph(self):
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        plt.figure(figsize=(8, 8))
        # for detection in self.hand_detections:
        #     handX = detection[0]
        #     handY = detection[1]
        #     plt.scatter(handX, handY, color=colors[3 % len(colors)], marker='o')

        bbox1, bbox2 = self.get_boundingbox()
        plt.plot(bbox1[0], bbox1[1], color=colors[1 % len(colors)], linestyle='dashed', label=f'Bounding Box left hand')
        plt.plot(bbox2[0], bbox2[1], color=colors[2 % len(colors)], linestyle='dashed', label=f'Bounding Box right hand')

        # Labeling axes
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(f'2D Coordinate System with all detections')

        # Set grid and limits for visibility
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

        # Reverse the y-axis
        plt.gca().invert_yaxis()

        # Add a legend to differentiate the balls
        plt.legend()

        # Display the plot
        plt.show()


if __name__ == '__main__':
    extractor = Siteswap_Graph_Plotter(max_balls=2)
    catch_history = [[2, 'left'], [1, 'right'], [2, 'left'], [1, 'right'], [1, 'right'], [3, 'left'], [2, 'right'], [2, 'right'], [1, 'left'], [3, 'right'], [3, 'right'], [2, 'left'], [1, 'right']]
    repeating_patterns = extractor.catch_history_patterns([1, 1, 2, 3, 2, 1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 2, 3, 2, 1, 2, 1, 3, 3, 2, 2, 2, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 3, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 1, 2, 1, 3, 3, 2, 2, 2, 1, 2, 1, 1, 3, 3, 2, 1, 2, 1, 3, 3, 3, 2, 2, 1, 1, 3, 3, 2, 2, 2, 1, 2, 1, 3, 1, 3, 2, 2, 2, 1, 1, 1, 3, 3, 2, 2, 1, 1, 3, 1, 3, 2, 2, 1, 1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 2, 2, 1, 2, 1, 3, 3, 3, 2, 3, 2, 1, 1, 3, 3, 3], 3)
    for pattern, count in repeating_patterns:
        print(f"Pattern: {pattern}, Repeats: {count}")