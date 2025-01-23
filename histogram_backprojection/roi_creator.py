import cv2
import numpy as np

class Roi_Creator:
    def __init__(self, image):
        self.__image = image
        self.center = None
        self.radius = None

    def __mouse_callback(self, event, x, y, flags, param):
        # Record the center and radius based on mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.center is None:
                self.center = (x, y)  # First click sets the center
                print(f"Center selected at: {self.center}")
            else:
                # Second click determines the radius
                self.radius = int(((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2) ** 0.5)
                print(f"Radius selected: {self.radius}")

    def create_circular_roi(self):
        # Resize the image for display purposes
        scale_factor = 0.5
        resized_image = cv2.resize(self.__image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Setup mouse callback
        cv2.namedWindow("Select Circle")
        cv2.setMouseCallback("Select Circle", self.__mouse_callback)

        print("Click to select the center of the circle, then click to define the radius.")
        while True:
            display_image = resized_image.copy()
            if self.center is not None:
                # Draw the circle dynamically if the radius is selected
                if self.radius is not None:
                    cv2.circle(display_image, self.center, self.radius, (0, 255, 0), 2)

            cv2.imshow("Select Circle", display_image)
            key = cv2.waitKey(1) & 0xFF

            # Press 'q' to exit after selecting the circle
            if key == ord('q') and self.center is not None and self.radius is not None:
                break

        cv2.destroyAllWindows()

        # Scale the center and radius back to the original image dimensions
        scaled_center = (int(self.center[0] / scale_factor), int(self.center[1] / scale_factor))
        scaled_radius = int(self.radius / scale_factor)

        # Create a mask for the circular ROI
        mask = np.zeros(self.__image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, scaled_center, scaled_radius, 255, -1)

        # Apply the mask to the original image
        cropped_image = cv2.bitwise_and(self.__image, self.__image, mask=mask)

        # Display the cropped image
        cv2.imshow("Cropped Circular ROI", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cropped_image
