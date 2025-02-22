import cv2
import numpy as np

def Daugman_normalization(img, pupil_params, iris_params):

    pupil_center, pupil_radius = pupil_params

    # Check if iris_params are valid
    if iris_params is None or iris_params[0] is None:
        print("Warning: Iris parameters not found, skipping normalization.")
        return None  # Or return an empty array as a fallback

    # Unpack iris parameters if they are valid
    iris_center, iris_radius = iris_params
    x_pupil, y_pupil = pupil_center
    x_iris, y_iris = iris_center

    # Polar coordinate mapping
    theta = np.linspace(0, 2 * np.pi, 512)
    r = np.linspace(pupil_radius, iris_radius, 64)
    normalized_iris = np.zeros((64, 512))

    h, w = img.shape

    # Map iris region into polar coordinates
    for i in range(512):
        for j in range(64):
            x = int(x_pupil + r[j] * np.cos(theta[i]))
            y = int(y_pupil + r[j] * np.sin(theta[i]))

            if 0 <= x < w and 0 <= y < h:
                normalized_iris[j, i] = img[y, x]
            else:
                normalized_iris[j, i] = 0

    return normalized_iris