

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial import distance
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding
from scipy.spatial.distance import cityblock, euclidean, cosine


from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *

import numpy as np
import random


# This function uses all modules before. For each fileName, first read in the file
# as image, then do Iris Localization, Normalization, Image Enhancement, and then
# extract features from that.
def select_test_sample(test_features, sample_size=108):
    """
    Randomly select a subset of test features and their classes with replacement

    Parameters:
    - test_features: np.array of test features
    - test_classes: np.array of test classes
    - sample_size: int, number of samples to select (default 108)

    Returns:
    - sample_features: np.array, randomly selected test features
    - sample_classes: np.array, corresponding classes of selected test features
    """
    test_classes = range(1, 110)
    # Randomly select indices with replacement
    indices = random.choices(test_classes, k=sample_size)
    sample_classes = []
    sample_features = []

    # Create dictionary to take one of the images from a given class, choosing
    # a different image if the class is repeated
    selected_dict = {}
    for i in range(1, 110):
      selected_dict[i] = np.array([1, 2, 3, 4])

    # Iterate, providing one of the images from each class
    for i in indices:
      random_index = np.random.choice(len(selected_dict[i]))
      selected_image = selected_dict[i][random_index]
      selected_dict[i] = np.delete(selected_dict[i], random_index)
    
      sample_classes.append(i)

      # Draw the right feature vector from test_features
      selected_image_fv = test_features[(i-1)*4 + selected_image]
      sample_features.append(selected_image_fv)
    
    return sample_features, sample_classes


def calculate_test(train_features, train_classes, test_sample, test_class, distance_type):
    """
    Calculate minimum distances with offsets for classification.

    Parameters:
    - train_features: np.array of training feature vectors
    - train_classes: np.array of training class labels
    - test_sample: np.array, single test feature vector
    - test_class: int, actual class of the test sample
    - distance_type: int, 1 for L1, 2 for L2, and 3 for Cosine

    Returns:
    - predicted_class: int, predicted class of the test sample
    - matched_dists: list, distances for matched samples
    - nonmatched_dists: list, distances for non-matched samples
    """
    dists = np.zeros(len(train_classes))
    matched_dists = []
    nonmatched_dists = []
    offset_range = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

    # Iterate through the training feature vectors, calculating distances from the test sample. 
    for i in range(len(train_classes)):
        dist_with_offset = np.ones(len(offset_range))

        for j, offset in enumerate(offset_range):
            shifted_sample = np.roll(test_sample, offset)
            
            # L1
            if distance_type == 1:
                dist_with_offset[j] = distance.cityblock(train_features[i, :], shifted_sample)
            # L2
            elif distance_type == 2:
                dist_with_offset[j] = distance.euclidean(train_features[i, :], shifted_sample)
            # Cosine
            elif distance_type == 3:  
                dist_with_offset[j] = distance.cosine(train_features[i, :], shifted_sample)

        # Minimum distance after applying offsets
        dists[i] = np.min(dist_with_offset)

        # Stick distances into matched and unmatched
        if train_classes[i] == test_class:
            matched_dists.append(dists[i])
        else:
            nonmatched_dists.append(dists[i])

    # Get predicted class based on the closest match
    predicted_class = train_classes[np.argmin(dists)]
    return predicted_class, matched_dists, nonmatched_dists


def match_iris(train_features, train_labels, test_features, test_labels, distance_type):
    """
    Calculate correct recognition rate (CRR) for a specified distance type.

    Parameters:
    - train_features: np.array of training feature vectors
    - train_labels: np.array of labels for training data
    - test_features: np.array of test feature vectors
    - test_labels: np.array of labels for test data
    - distance_type: int, 1 for L1, 2 for L2, and 3 for Cosine

    Returns:
    - cr_rate: CRR for the specified distance type
    - matched_dists: distances for correctly matched samples
    - nonmatched_dists: distances for incorrect matches
    """
    matched_dists, nonmatched_dists = [], []
    correct = 0.0
    total = len(test_labels)

    for idx, test_vector in enumerate(test_features):
        # Calculate distances
        distances = []
        for train_vector in train_features:
            if distance_type == 1:
                distances.append(cityblock(test_vector, train_vector))
            elif distance_type == 2:
                distances.append(euclidean(test_vector, train_vector))
            elif distance_type == 3:
                distances.append(cosine(test_vector, train_vector))

        # Identify closest match
        closest_idx = np.argmin(distances)
        if train_labels[closest_idx] == test_labels[idx]:
            correct += 1
            matched_dists.append(distances[closest_idx])
        else:
            nonmatched_dists.append(distances[closest_idx])

    cr_rate = (correct / total) * 100

    # Identify the closest centroid
    #    predicted_label = min(distances, key=distances.get)

    #    if predicted_label == test_labels[idx]:
    #        correct += 1
    #        matched_dists.append(distances[predicted_label])
    #    else:
    #        nonmatched_dists.append(distances[predicted_label])

    return cr_rate, matched_dists, nonmatched_dists

def match_with_reduction(train_features, train_labels, test_features, test_labels, num_components):
    """
    Perform dimensionality reduction on features and compute CRR for three distance types.

    Parameters:
    - train_features: np.array of training feature vectors
    - train_labels: np.array of labels for training data
    - test_features: np.array of test feature vectors
    - test_labels: np.array of labels for test data
    - num_components: int, desired dimensionality after reduction

    Returns:
    - cr_l1, cr_l2, cr_cosine: CRRs for L1, L2, and Cosine measures
    """
    n_features = train_features.shape[1]
    n_classes = len(np.unique(train_labels))
    max_components = min(n_features, n_classes - 1)

    if num_components < max_components:
        reducer = LDA(n_components=num_components)
        train_reduced = reducer.fit_transform(train_features, train_labels)
        test_reduced = reducer.transform(test_features)
    # If num_components exceeds max allowable, use max_components
    else:
        reducer = LDA(n_components=max_components)
        train_reduced = reducer.fit_transform(train_features, train_labels)
        test_reduced = reducer.transform(test_features)

    # Use KNN with different distances metrics
    cr_l1 = knn_recognition_rate(train_reduced, train_labels, test_reduced, test_labels, 'manhattan')
    cr_l2 = knn_recognition_rate(train_reduced, train_labels, test_reduced, test_labels, 'euclidean')
    cr_cosine = knn_recognition_rate(train_reduced, train_labels, test_reduced, test_labels, 'cosine')

    return cr_l1, cr_l2, cr_cosine

def knn_recognition_rate(train_reduced, train_labels, test_reduced, test_labels, metric):
    """
    Compute CRR with KNN for a specified metric.
    """
    knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
    knn.fit(train_reduced, train_labels)
    predictions = knn.predict(test_reduced)
    return (np.sum(predictions == test_labels) / len(test_labels)) * 100

def bootstrap_matching(train_features, train_labels, test_features, test_labels, repetitions, threshold_list):
    """
    Perform CRR bootstrap evaluation on different thresholds.

    Parameters:
    - repetitions: repetitions
    - threshold_list: thresholds for FMR and FNMR
    """
    all_fmrs, all_fnmrs = [], []
    cr_rates = np.zeros(repetitions)

    # Reduce dimensions using LLE
    lle = LocallyLinearEmbedding(n_neighbors=201, n_components=200)
    train_reduced = lle.fit_transform(train_features)
    test_reduced = lle.transform(test_features)

    # Perform the specified number of repetitions, sampling randomly from the test set and calculating metrics
    for rep in range(repetitions):
        # Sample
        sampled_features, sampled_labels = select_random_sample(test_reduced, test_labels)

        # Calculate
        cr_rate, dist_match, dist_nonmatch = match_iris(train_reduced, train_labels, sampled_features, sampled_labels, 3)
        fmrs, fnmrs = calcROC(dist_match, dist_nonmatch, threshold_list)

        # Store
        all_fmrs.append(fmrs)
        all_fnmrs.append(fnmrs)
        cr_rates[rep] = cr_rate

    # Calculate mean, std, upper and lower bounds
    cr_mean = np.mean(cr_rates)
    cr_std = np.std(cr_rates)
    cr_upper = min(cr_mean + cr_std * 1.96, 100)
    cr_lower = max(cr_mean - cr_std * 1.96, 0)

    return np.array(all_fmrs), np.array(all_fnmrs), cr_mean, cr_upper, cr_lower

def select_random_sample(test_reduced, test_labels):
    """
    Select a random sample for testing from test set.
    """
    indices = np.random.choice(len(test_labels), size=108, replace=False)
    sampled_features = test_reduced[indices]
    sampled_labels = np.array([test_labels[i] for i in indices])
    return sampled_features, sampled_labels

# Function to calculate CRR across different distance measures
def compute_crr_summary(train_features, train_labels, test_features, test_labels):
    """
    Calculate and display the CRR for feature vectors.
    Includes L1, L2, and Cosine distances.
    """
    # Calculate CRRs for original feature set
    l1_crr_original, _, _ = match_iris(train_features, train_labels, test_features, test_labels, distance_type=1)
    l2_crr_original, _, _ = match_iris(train_features, train_labels, test_features, test_labels, distance_type=2)
    cosine_crr_original, matched_dists, nonmatched_dists = match_iris(train_features, train_labels, test_features, test_labels, distance_type=3)

    # Generate table only if values are valid
    if not (l1_crr_original and l2_crr_original and cosine_crr_original):
        print("CRR value None")
        return

   # Calculate CRRs for reduced feature set with dimension 200
    l1_crr_reduced, l2_crr_reduced, cosine_crr_reduced = match_with_reduction(train_features, train_labels, test_features, test_labels, num_components=200)

    # Display results in a table format
    print("Recognition Rate Summary (%)")
    print(tabulate([
        ['L1 distance', l1_crr_original, l1_crr_reduced ],
        ['L2 distance', l2_crr_original, l2_crr_reduced ],
        ['Cosine similarity', cosine_crr_original, cosine_crr_reduced]
    ], headers=['Distance Measure', 'Original Set', 'Reduced Set']))

    default_thresholds = np.arange(0.04, 0.1, 0.003)

    # Generate and display ROC curve for cosine similarity
    fmrs, fnmrs = calcROC(matched_dists, nonmatched_dists, default_thresholds)
    plot_roc_curve(fmrs, fnmrs, 'roc_curve.png')

def calcROCBootstrap(fmrs, fnmrs):
    """
    Calculate the mean, lower (5th percentile), and upper (95th percentile) bounds
    for FMR and FNMR using bootstrapping.

    Parameters:
    - fmrs: np.array, bootstrap samples of FMRs
    - fnmrs: np.array, bootstrap samples of FNMRs

    Returns:
    - Tuple of means, lower bounds, and upper bounds for FMR and FNMR
    """
    fmrs_mean = np.mean(fmrs, axis=0)
    fmrs_l = np.percentile(fmrs, 5, axis=0)
    fmrs_u = np.percentile(fmrs, 95, axis=0)

    fnmrs_mean = np.mean(fnmrs, axis=0)
    fnmrs_l = np.percentile(fnmrs, 5, axis=0)
    fnmrs_u = np.percentile(fnmrs, 95, axis=0)

    return fmrs_mean, fmrs_l, fmrs_u, fnmrs_mean, fnmrs_l, fnmrs_u

def calcROC(matched_dists, nonmatched_dists, thresholds):
    """
    Calculate FMR and FNMR for each threshold.

    Parameters:
    - matched_dists: np.array, distances for correctly matched pairs
    - nonmatched_dists: np.array, distances for incorrectly matched pairs
    - thresholds: list, thresholds for calculating FMR and FNMR

    Returns:
    - fmrs: list, FMR for each threshold
    - fnmrs: list, FNMR for each threshold
    """
    matched_dists = np.array(matched_dists)
    nonmatched_dists = np.array(nonmatched_dists)
    n_m = len(matched_dists)
    n_nm = len(nonmatched_dists)

    # Initialize FMR and FNMR lists
    fmrs = []
    fnmrs = []

    # Calculate FMR and FNMR for each threshold
    for t in thresholds:
        fmr = np.sum(nonmatched_dists < t) / n_nm
        fnmr = np.sum(matched_dists > t) / n_m

        fmrs.append(fmr)
        fnmrs.append(fnmr)

    return fmrs, fnmrs