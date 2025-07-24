import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform, color
from skimage.feature import hog, SIFT, match_descriptors
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

# TODO: Create feature processing functions for SIFT and HOG
from evaluate_sift import extract_sift_features
from evaluate_hog import extract_hog_features

if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # TODO: Extract features from the training data
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.uint8)
    y_test = data["y_test"]

    print(X_train.shape, y_train.shape)
    print(X_train.dtype, y_train.dtype)
    print(X_test.shape, y_test.shape)
    print(X_test.dtype, y_test.dtype)
    # use SIFT to extract features

    x_train_sift, y_train_sift, svm, kmeans, tfidf = extract_sift_features(
        X_train, y_train, train=True)
    x_train_hog, y_train_hog, svm_hog, kmeans_hog, tfidf_hog = extract_hog_features(
        X_train, y_train, train=True)
    # TODO: Extract features from the testing data
    x_test_sift, y_test_sift = extract_sift_features(
        X_test, y_test, train=False,
        svm=svm,
        kmeans=kmeans,
        tfidf=tfidf)
    x_test_hog, y_test_hog = extract_hog_features(
        X_test, y_test, train=False,
        svm=svm_hog,
        kmeans=kmeans_hog, tfidf=tfidf_hog)
    # TODO: Save the extracted features to a file
    # np.savez('cifar10_sift_bow.npz',
    #          X_train=x_train_sift,
    #          y_train=y_train_sift,
    #          X_test=x_test_sift,
    #          y_test=y_test_sift)
    # np.savez('cifar10_hog_bow.npz',
    #          X_train=x_train_hog,
    #          y_train=y_train_hog,
    #          X_test=x_test_hog,
    #          y_test=y_test_hog)

# SIFT IS BETTER THAN HOG DUE TO invariant to rotation, scale and translation, they can capture more complex features of the same class
# HOG is more suitable for object detection and recognition tasks where the objects are not rotated or scaled, it is also faster to compute than SIFT
# BUT IN THIS ASSIGNMENT, HOG IS BETTER THAN SIFT DUE TO THE SMALL SIZE OF THE IMAGES (32x32) its hard to draw keypoints and descriptors from low resolution images
