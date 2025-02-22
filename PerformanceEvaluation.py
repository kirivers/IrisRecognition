import numpy as np
import matplotlib.pyplot as plt

from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *
from IrisMatching import match_iris, match_with_reduction

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Thresholds for ROC

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
# Plot ROC curve
def plot_roc_curve(fmrs, fnmrs, filename):
    plt.plot(fmrs, fnmrs, label='ROC Curve')
    plt.xlabel('FMR')
    plt.ylabel('FNMR')
    plt.title('ROC Curve for Iris Matching')
    plt.savefig(filename)
    plt.show()

