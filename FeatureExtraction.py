import cv2
import numpy as np


def rotate_image(img, r, pupil_center):
  """
  Rotates the provided image by r degrees around the center of the pupil
  """
  # Get dimensions
  h, w = img.shape[:2]

  pc0 = pupil_center[0]
  pc1 = pupil_center[1]
  if type(pupil_center[0]) != int:
    pc0 = pupil_center[0].item()
    pc1 = pupil_center[1].item()

  # Get rotation matrix
  rotation_matrix = cv2.getRotationMatrix2D((pc0, pc1), r, 1.0)
  # Apply rotation matrix
  img_rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
  return img_rotated

def crop_ROI(img):
  """
  Crops the image to an upper portion.
  Returns the cropped image.
  """
  h, w = img.shape[:2]
  new_h = h * 4 // 5
  return img[:new_h, :]

def apply_defined_filter(img, d_x, d_y):
  """
  Apply the defined filter:
  G(x, y, f) = 1/(2π∂_x∂_y)exp[-1\2( x^2/(∂_x)^2) + y^2/(∂_y^2) )]M_1(x, y, f);
  M_1(x, y, f) = cos[ 2πf * sqrt(x^2 + y^2) ]
  Where f is the frequency of the sinusoidal function,
  ∂_x, ∂_y are the space constants of the Gaussian envelope along the x and y axis, respectively
  and θ denotes the orientation of the defined filter.
  Returns the filtered image
  """
  #theta = np.pi/2
  theta = 0
  f = 0.3

  x = np.linspace(-15, 15, 31)
  y = np.linspace(-15, 15, 31)
  X, Y = np.meshgrid(x, y)

  kernel = 1/(2 * np.pi * d_x * d_y) * np.exp(-(X**2 / d_x**2 + Y**2 / d_y**2)) * np.cos(2 * np.pi * f * np.sqrt(X**2 + Y**2))

  # Apply defined filter
  filtered_img = cv2.filter2D(img, cv2.CV_32F, kernel)

  return filtered_img

def extract_feature_vector(img):
  """
  Extract:
  m = 1/n ∑ |F_i(x, y)|
  σ = 1/n ∑ ||F_i(x, y)|-m|
  for each 8x8 region, applied to both images.
  Where n is the number of pixels in the block.
  Return the vector V = [m_1, σ_1,...]
  """
  block_size = 8
  h, w = img.shape[:2]

  # Initialize feature vector
  feature_vector = []

  # For each block, calculate mean and std
  for i in range(0, h, block_size):
      for j in range(0, w, block_size):
          block = img[i:i + block_size, j:j + block_size]
          # Mean
          m = np.mean(np.abs(block))
          # Std
          s = np.mean(np.abs(np.abs(block) - m))
          feature_vector.append(m)
          feature_vector.append(s)

  return np.array(feature_vector)