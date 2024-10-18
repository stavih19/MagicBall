import time
import cv2
import numpy as np
import os
import joblib
from CutCircles import cut_circles
from SvcClassifier import svc_interference


def mark_balls(dir_origin_path, dir_detect_path, img_name):
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

    cutted_circles = cut_circles(circles, image, gray)
    svc_classifier = joblib.load('svc_model.pkl')
    predictions = []
    for circle in cutted_circles:
        s = circle.shape
        if s[0] < 1 or s[1] < 1 or s[2] < 1:
            continue
        predictions.append(svc_interference(circle, svc_classifier))

    max_value = max(predictions)
    max_index = predictions.index(max_value)

    simple_circles = []
    # print(circles[0])
    for index in range(circles.shape[1]):
        if max_value - predictions[index] < 0.2 and predictions[index] > 0.3:
            circle = np.round(circles[0][index]).astype("int")
            if abs(circle[2] -+ circles[0][max_index][2]) < 20:
                simple_circles.append((circle[0], circle[1], circle[2]))
    circles = simple_circles
    predictions.sort(reverse=True)
    # print(predictions)

    # Step 6: Draw the detected circles and edges
    distance = []
    if circles is not None:
        for (x, y, r) in circles:
            # Draw the circle
            cv2.circle(image, (x, y), r + 1, (0, 255, 0), 1)

            perceived_radius = r  # Radius in pixels from the image
            distance.append((800 * 10) / perceived_radius)
            # print(f'distance: {distance}')

    else:
        print(f'Not found {img_name}')
        return

    # Step 7: Overlay edges on the original image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges to 3 channels
    output = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

    # Create the output path for the processed image
    output_image_path = os.path.join(dir_detect_path, img_name + '_distance: ' + str('|'.join(distance)))
    cv2.imwrite(output_image_path, output)

    # Step 8: Show the result
    cv2.imshow('Detected Balls with Edges', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Path to the folder containing images
folder_origin_path = 'D:\\PythonProjects\\MagicBall\\Find the ball\\Balls\\Origin'
folder_detect_path = 'D:\\PythonProjects\\MagicBall\\Find the ball\\Balls\\Detection'

# List all files in the folder
image_files = [f for f in os.listdir(folder_origin_path) if f.endswith('.JPG')]

start_time = time.time()
for image_name in image_files[0:]:
    mark_balls(folder_origin_path, folder_detect_path, image_name)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")