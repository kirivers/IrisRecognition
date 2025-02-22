import cv2
import numpy as np

def get_background_estimation(img):
  """
  Finds the mean of each 16x16 block and expands it
  to the same size as the normalized image by
  bicubic interpolation.
  """

  # downsampling block size
  block_size = 16
  h, w = img.shape[:2]

  # resize image where each pixel represents a 16x16 block in bigger image (mean)
  small_img = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_AREA)

  # resize to normal dimensions for background smoothing
  background_estimation = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_CUBIC)
  return background_estimation

def apply_background(img, background_estimation):
  """
  Takes the difference between the image and the
  background estimation, returning a clearer image.
  """

  # subtract background estimation from original image
  enhanced_img = cv2.subtract(img, background_estimation)
  return enhanced_img

def enhance_image(img):
  """
  Enhance the lighting of the new image via histogram
  equalization in each 32x32 region.
  """

  # image block size
  block_size = 32
  h, w = img.shape[:2]
  enhanced_img = np.zeros_like(img)

  # loop over each block
  for i in range(0, h, block_size):
      for j in range(0, w, block_size):
          # extract block
          block = img[i:i+block_size, j:j+block_size]
          #gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
          #enhanced_block = cv2.equalizeHist(gray_block)

          # histogram equalization to enhance image contrast within block
          enhanced_block = cv2.equalizeHist(block.astype(np.uint8))
          #enhanced_img[i:i+block_size, j:j+block_size] = cv2.cvtColor(enhanced_block, cv2.COLOR_GRAY2BGR)

          # replace original block with enhanced contrast block
          enhanced_img[i:i+block_size, j:j+block_size] = enhanced_block
  return enhanced_img