import argparse
import cv2
import time

from ball_tracking.ball_detector import Ball_Detector
from ball_tracking.ball_tracker import Ball_Tracker
from catch_detector import Catch_Detector
from hand_tracking.hand import Hand
from hand_tracking.hand_detector import Hand_Detector
from hand_tracking.hand_tracker import Hand_Tracker
from roi_creator import Roi_Creator
from siteswap_graph_plotter import Siteswap_Graph_Plotter
from siteswap_predictor import Siteswap_predictor

# get launch arguments
def get_launch_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-b", "--max_balls", default=1, help="max number of balls")
    ap.add_argument("-o", "--output", help="path to the output video file", default="output.mp4")
    args = vars(ap.parse_args())

    if not args.get("video", False):
        return {
            "video_path": "C:/Uni/Abschlussarbeit/Siteswaps/singleBallTracking/samples/bht/ss441_id_01.mp4",
            "max_balls": 3,
            "output_path": "output.mp4"
        }
    
    return {
        "video_path": args["video"],
        "max_balls": int(args["max_balls"]),
        "output_path": args["output"]
    }

def display_frame(title: str, frame):
    displayed_frame = frame.copy()
    displayed_frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(title, displayed_frame)

args = get_launch_arguments()
video_path = args["video_path"]
max_balls = args["max_balls"]
output_path = args["output_path"]

cap = cv2.VideoCapture(video_path)
ball_tracker = Ball_Tracker(max_balls)
fps = 1
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize VideoWriter
fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

siteswap_graph_plotter = Siteswap_Graph_Plotter(max_balls)

left_hand = Hand(0)
right_hand = Hand(1)
hand_detector = Hand_Detector()
hand_tracker = Hand_Tracker(left_hand, right_hand)
catch_detector = Catch_Detector(max_balls, left_hand, right_hand)

siteswap_predictor = Siteswap_predictor(max_balls)

ret, frame = cap.read()

roi_creator = Roi_Creator(frame)
roi = roi_creator.create_circular_roi()

ball_detector = Ball_Detector(max_balls, roi)

background_subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=20, detectShadows=False)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    original_frame = frame.copy()

    color_mask = ball_detector.remove_other_colors(frame)

    fg_mask = background_subtractor.apply(frame)
    
    blurred_mask = cv2.GaussianBlur(fg_mask, (9, 9), 2)

    blurred_mask = cv2.erode(blurred_mask, None, iterations=2)
    blurred_mask = cv2.dilate(blurred_mask, None, iterations=2)

    combined_mask = cv2.bitwise_and(color_mask, blurred_mask)

    contours = ball_detector.get_max_contours(combined_mask)

    if not contours:
        display_frame("Frame", frame)
        display_frame("Color Mask", combined_mask)
        out.write(frame)
        key = cv2.waitKey(fps)
        if key == ord("q"):
            break
        elif key == ord(" "):
            continue
        continue

    # detect balls
    detections = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        detections.append([x, y])
        center: cv2.typing.Point = (int(x), int(y))

        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    # detect hands
    hands_center_id = []
    hand_detections = hand_detector.detect_hands(original_frame)
    for x, y in hand_detections:
        cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)

    # Update the tracker with the current frames detections
    balls_center_id = ball_tracker.update(detections)
    siteswap_graph_plotter.add_ball_detections(balls_center_id)
    
    hand_tracker.update(hand_detections)
    if (len(left_hand.get_coords()) > 0 and len(right_hand.get_coords()) > 0):
        left_x, left_y = left_hand.get_coords()
        right_x, right_y = right_hand.get_coords()
        siteswap_graph_plotter.add_hand_detections([[left_x, left_y, left_hand.get_id()], [right_x, right_y, right_hand.get_id()]])

    # display tracked ball detections
    for ball_center_id in balls_center_id:
        x, y, id = ball_center_id
        center = (int(x), int(y))
        cv2.putText(frame, str(id), (int(x), int(y) - 15), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # display tracked hand detections
    area = 20000
    hand_bboxes = []
    if (len(left_hand.get_coords()) > 0 and len(right_hand.get_coords()) > 0):
        xl, yl = left_hand.get_coords()
        xr, yr = right_hand.get_coords()

        cv2.circle(frame, (xl, yl), 15, (0, 255, 0), -1)
        cv2.circle(frame, (xr, yr), 15, (255, 0, 0), -1)

        top_left_0, bottom_right_0 = left_hand.get_bbox()
        top_left_1, bottom_right_1 = right_hand.get_bbox()
        cv2.rectangle(frame, top_left_0, bottom_right_0, 15)
        cv2.rectangle(frame, top_left_1, bottom_right_1, 15)
        hand_bboxes.append(tuple([top_left_0, bottom_right_0]))
        hand_bboxes.append(tuple([top_left_1, bottom_right_1]))

    catch_detector.update(balls_center_id)
    temp_history = catch_detector.get_catch_history()
    # last 10 entries
    temp_history = temp_history[-10:]
    cv2.putText(frame, str(temp_history), (frame.shape[1] - 1000, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, bottomLeftOrigin=False)

    # show frames
    display_frame("Frame", frame)
    display_frame("Color Mask", combined_mask)

    # Write the frame to the output video
    out.write(frame)

    key = cv2.waitKey(fps)
    if key == ord("q"):
        break
    elif key == ord(" "):
        continue

siteswap_graph_plotter.plot_pattern_graph()
catch_history = catch_detector.get_catch_history()
siteswaps = siteswap_predictor.predict_possible_siteswaps(catch_history)
siteswaps_with_confidence = siteswap_predictor.calculate_confidence(siteswaps)
print("Possible siteswaps", siteswaps_with_confidence)

cap.release()
out.release()
cv2.destroyAllWindows()