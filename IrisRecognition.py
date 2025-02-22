import os
import cv2
import numpy as np

from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *
from PerformanceEvaluation import *

image_folder = 'Images'

# Set paths and parameters
rotation_angles = [-9, -6, -3, 0, 3, 6, 9]

# Initialize lists for image paths and labels
image_paths_train, image_paths_test = [], []

# Load images and labels
for folder_num in range(1, 109):  # Assuming 108 subjects
    folder_name = f"{folder_num:03d}"
    folder_path = os.path.join(image_folder, folder_name)

    # Training (session 1) and testing (session 2) images
    train_subfolder = os.path.join(folder_path, "1")
    test_subfolder = os.path.join(folder_path, "2")

    # Append paths to train and test lists
    for img_file in os.listdir(train_subfolder):
        image_paths_train.append(os.path.join(train_subfolder, img_file))
    for img_file in os.listdir(test_subfolder):
        image_paths_test.append(os.path.join(test_subfolder, img_file))

# Function to preprocess images and extract features
# Function to preprocess images and extract features
def process_image(file_path, rotation=None):
    # Load the image in grayscale
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if img is None:
        print(f"Warning: Image {file_path} could not be loaded.")
        return None

    # Step 1: Localization
    pupil_center = get_pupil_center(img)
    pupil_params = get_pupil_radius(img, pupil_center)
    pupil_params, iris_params = get_pupil_iris_location(img, pupil_params[0], pupil_params[1])

    # Rotate image if specified
    if rotation:
        img = rotate_image(img, rotation, pupil_params[0])

    # Step 2: Normalization
    normalized_iris = Daugman_normalization(img, pupil_params, iris_params)
    if normalized_iris is None:
        print(f"Warning: Normalization failed for image {file_path}.")
        return None  # Skip if normalization fails

    # Step 3: Enhancement
    bg = get_background_estimation(normalized_iris)
    img_enhanced = apply_background(normalized_iris, bg)
    img_enhanced = enhance_image(img_enhanced)

    # Step 4: Feature Extraction
    roi = crop_ROI(img_enhanced)
    filtered_1 = apply_defined_filter(roi, 3, 4.5)
    filtered_2 = apply_defined_filter(roi, 1.5, 1.5)
    fv_1 = extract_feature_vector(filtered_1)
    fv_2 = extract_feature_vector(filtered_2)

    return np.concatenate((fv_1, fv_2))

# Extract feature vectors
train_features, test_features = [], []
train_labels, test_labels = [], []

rotation_angles = [-9, -6, -3, 0, 3, 6, 9]

print("training...")
for img_path in image_paths_train:
    print("img_path: ", img_path)
    # Only process BMP files
    if img_path.lower().endswith('.bmp'):
        for angle in rotation_angles:
            feature_vector = process_image(img_path, rotation=angle)
            if feature_vector is not None:
                train_features.append(feature_vector)
                train_labels.append(img_path.split('/')[-3])  # Label by folder number

print("testing...")
for img_path in image_paths_test:
    print("img_path: ", img_path)
    # Only process BMP files
    if img_path.lower().endswith('.bmp'):
        feature_vector = process_image(img_path)
        if feature_vector is not None:
            test_features.append(feature_vector)
            test_labels.append(img_path.split('/')[-3])  # Label by folder number

# Convert to arrays
train_features = np.array(train_features)
test_features = np.array(test_features)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Generate Tables and Plots
print("Correct recognition rate table")
compute_crr_summary(train_features, train_labels, test_features, test_labels)

# Bootstrapping for ROC
repetitions = 100
thresholds = np.linspace(0.04, 0.1, 10)
fmrs, fnmrs, crr_mean, crr_upper, crr_lower = bootstrap_matching(
    train_features, train_labels, test_features, test_labels, repetitions, thresholds
)