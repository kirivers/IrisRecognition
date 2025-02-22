import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_pupil_center(image):
  """
  Returns: the center coordinates

  Projects the image in the vertical and horizontal
  direction to approximately estimate the center
  coordinates (Xp, Yp) of the pupil. The minima of
  the two projection profiles are considered as the
  center coordinates of the pupil.
  """

  #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = image.copy()
  h, w = img.shape[:2]

  # White edges
  img[:, :60] = 255
  img[:, w-60:] = 255
  img[:90, :] = 255
  img[h-60:, :] = 255

  # vertical projection obtained by summing along rows (axis = 1)
  vertical_projection = np.sum(img, axis=1)

  # horizontal projection obtained by summing along columns (axis = 0)
  horizontal_projection = np.sum(img, axis=0)

  # center coordinates will be minima of projection profiles
  Yp = np.argmin(vertical_projection)
  Xp = np.argmin(horizontal_projection)
  return (Xp, Yp)

def get_pupil_radius(img, center):
  """
  Returns: more accurate center coordinates, radius
  of the pupil (in pixels)

  Binarize a 120x120 region around the center using
  a threshold from a gray-level histogram. The
  centroid estimates the pupils coordinates, and
  we can also calculate the radius.

  """
  x_center, y_center = center

  # measure of distance from center
  exp = 60

  # crop a 120x120 region around the center of the pupil
  region = img[y_center-exp:y_center+exp, x_center-exp:x_center+exp]
  #gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

  # binary thresholding with Otsu's method to segment the pupil area
  _, binary_region = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # contours in the binary image to identify pupil boundary
  contours, _ = cv2.findContours(binary_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
      # assuming the largest contour in the defined region is the pupil
      largest_contour = max(contours, key=cv2.contourArea)

      # fitting a circle (since pupil is circular) to this largest contour
      ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
      refined_center = (int(x + x_center - 60), int(y + y_center - 60))
      radius = int(radius)
  else:
      refined_center = center
      radius = 0

  return refined_center, radius

def get_pupil_iris_location(image, center, pupil_radius):

  # gaussian blur to reduce noise for better edge detection
  img = cv2.GaussianBlur(image, (7, 7), sigmaX = 1.5, sigmaY = 1.5)

  x_center, y_center = center
  # region of interest radius for pupil localization
  roi_d = int(pupil_radius)

  # associated coordinates for roi in image
  y1 = max(0, y_center - roi_d)
  y2 = min(img.shape[0], y_center + roi_d)
  x1 = max(0, x_center - roi_d)
  x2 = min(img.shape[1], x_center + roi_d)
  roi = img[y1:y2, x1:x2]

  # canny edge detection
  edges = cv2.Canny(roi, 50, 150)

  # hough circle transform to detect circles (pupil candidate) in edges
  circles = cv2.HoughCircles(
      edges,
      cv2.HOUGH_GRADIENT,
      dp=1,
      minDist=pupil_radius,
      param1=10,
      param2=15,
      minRadius=int(pupil_radius * 0.4),
      maxRadius=int(pupil_radius * 1)
    )

  if circles is not None:
      circles = np.round(circles[0, :]).astype("int")

      # assuming pupil is the smallest detected circle
      pupil_circle = min(circles, key=lambda c: np.hypot(c[0] + x1 - x_center, c[1] + y1 - y_center))

      # convert pupil coordinates into associated location in overall image
      pupil_center = (pupil_circle[0] + x1, pupil_circle[1] + y1)
      pupil_radius = pupil_circle[2]
  else:
      pupil_center, pupil_radius = center, pupil_radius

  # Same process for iris localization - bigger area around pupil so bigger roi
  roi_x = int(pupil_radius * 3)
  roi_y = int(pupil_radius * 3)
  y1 = max(0, pupil_center[1] - roi_y)
  y2 = min(img.shape[0], pupil_center[1] + roi_y)
  x1 = max(0, pupil_center[0] - roi_x)
  x2 = min(img.shape[1], pupil_center[0] + roi_x)

  img_iris = img.copy()

  # Apply histogram equalization
  scaled_img = cv2.convertScaleAbs(img_iris, alpha=1.5, beta=6)

  roi = scaled_img[y1:y2, x1:x2]
  roi = cv2.GaussianBlur(roi, (5, 5), 0)
  
  median = np.median(roi)

  # Set the thresholds
  lower_thres = int(median * 0.2)
  upper_thres = int(median * 0.4)

  # apply Canny edge detection with thresholds
  iris_edges = cv2.Canny(roi, lower_thres, upper_thres)

  # hough transform to detect iris in the roi
  circles = cv2.HoughCircles(
      iris_edges,
      cv2.HOUGH_GRADIENT,
      dp=1,
      minDist=pupil_radius * 2,
      param1=50,
      param2=10,
      minRadius=int(pupil_radius * 2),
      maxRadius=int(pupil_radius * 2.8)
    )


  if circles is not None:
      circles = np.round(circles[0, :]).astype("int")

      # assuming iris is the next smallest circle surrounding the pupil
      iris_circle = min(circles, key=lambda c: np.hypot(c[0] + x1 - pupil_center[0], c[1] + y1 - pupil_center[1]))

      # convert roi coordinates to larger image coordinates
      iris_center = (iris_circle[0] + x1, iris_circle[1] + y1)
      iris_radius = iris_circle[2]
      return (pupil_center, pupil_radius), (iris_center, iris_radius)
  else:
      return (pupil_center, pupil_radius), (pupil_center, pupil_radius*2)

