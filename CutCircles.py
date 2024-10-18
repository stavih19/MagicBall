import cv2
import numpy as np


def cut_circles(circles, image, gray):
    circles = np.uint16(np.around(circles))

    cropped_circles = []
    for circle in circles[0, :]:
        x, y, radius = circle

        # Create a mask for the circle
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), radius, 255, -1)  # Fill the circle in the mask

        # Extract the circle from the original image using the mask
        circle_image = cv2.bitwise_and(image, image, mask=mask)

        # Crop the circle image to get the bounding box around the circle
        x_min = max(0, int(x) - int(radius))
        x_max = min(image.shape[1], x + radius)
        y_min = max(0, int(y) - int(radius))
        y_max = min(image.shape[0], y + radius)

        cropped_circle = circle_image[y_min:y_max, x_min:x_max]
        cropped_circles.append(cropped_circle)
    return cropped_circles
