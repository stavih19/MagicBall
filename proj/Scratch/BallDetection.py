import time
import cv2
import numpy as np
import os
import joblib
from CutCircles import cut_circles
from SvcClassifier import svc_interference
from decimal import Decimal, ROUND_HALF_UP


def mark_balls(dir_origin_path, dir_detect_path, img_name):
    start_time = time.time()
    # Step 1: Load the image
    image = cv2.imread(os.path.join(dir_origin_path, img_name))

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur
    sigmaX = 2
    ksize = 9
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX, sigmaY=sigmaX)

    # Step 4: Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 200)

    # Step 5: Detection circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=40, param2=30, minRadius=15, maxRadius=70)

    # step 6: cut the detection circles from the image
    cutted_circles = cut_circles(circles, image, gray)
    predictions = []

    # step 7: predict if the circle is the one we're looking for
    for circle in cutted_circles:
        s = circle.shape
        if s[0] < 1 or s[1] < 1 or s[2] < 1:
            continue
        predictions.append(svc_interference(circle, svc_classifier))

    max_value = max(predictions)
    max_index = predictions.index(max_value)

    # step 8: collect all the possible circles that predicted as out ball
    simple_circles = []
    for index in range(circles.shape[1]):
        if max_value - predictions[index] < 0.2 and predictions[index] > 0.3:
            circle = np.round(circles[0][index]).astype("int")
            if abs(circle[2] - circles[0][max_index][2]) < 20:
                simple_circles.append((circle[0], circle[1], circle[2]))
    circles = simple_circles

    # Step 9: Draw the detected circles and edges
    distances = []
    if circles is not None:
        for (x, y, r) in circles:
            # Draw the circle
            cv2.circle(image, (x, y), r + 5, (0, 255, 0), 1)

            # Radius in pixels from the image
            pixel_diameter = r  # Diameter of the ball in pixels
            actual_diameter = 10  # Actual diameter of the ball in meters (10 cm)

            # Camera parameters (assume focal length is known)
            focal_length = 800  # Example focal length in pixels

            # Calculate the distance from the camera to the ball
            distance = (focal_length * actual_diameter) / pixel_diameter
            distances.append(round(distance))

    else:
        print(f'Not found {img_name}')
        return

    # Step 10: Overlay edges on the original image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3 channels
    output = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

    end_time = time.time()

    # Create the output path for the processed image
    distances = '-'.join(map(str, distances))
    time_duration = end_time - start_time
    time_duration_decimal = Decimal(str(time_duration))
    elapsed_time = time_duration_decimal.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
    output_image_path = os.path.join(dir_detect_path, img_name.split('.')[0] + '_' + str(elapsed_time) + '_.JPG')
    cv2.imwrite(output_image_path, output)


def run_detection(image_files, folder_origin_path, folder_detect_path):
    # interference
    start_time = time.time()
    for image_name in image_files[0:]:
        mark_balls(folder_origin_path, folder_detect_path, image_name)
    end_time = time.time()

    print(f'elapsed_time: {end_time - start_time}')
    return end_time - start_time


if __name__ == "__main__":
    current_directory = os.getcwd()

    # Load weights model
    svc_classifier = joblib.load(os.path.join(current_directory, 'proj', 'Scratch', 'svc_model.pkl'))

    # Path to the folder containing images
    folder_origin_path = os.path.join(current_directory, 'proj', 'Ball')
    folder_detect_path = os.path.join(current_directory, 'proj', 'Results')

    # List all files in the folder
    image_files = [f for f in os.listdir(folder_origin_path) if
                   (f.endswith('.JPG') or f.endswith('.jpg') or f.endswith('.png'))]

    elapsed_time_vec = []
    for i in range(10):
        elapsed_time_vec.append(run_detection(image_files, folder_origin_path, folder_detect_path))
    print(elapsed_time_vec)
    print(sum(elapsed_time_vec) / len(elapsed_time_vec))
