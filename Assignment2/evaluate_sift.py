from skimage.feature import SIFT
from skimage import color
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.svm import LinearSVC


def extract_sift_features(X, y, train=True, svm=None, kmeans=None, tfidf=None, vocab_size=100):
    """
    Extract SIFT features from images and train or test an SVM classifier.

    Args:
        X: Images (numpy array).
        y: Labels for the images (numpy array).
        train: If True, train the model. If False, test using provided models.
        svm: Trained SVM model (required if train=False).
        kmeans: Trained KMeans model (required if train=False).
        tfidf: Trained TfidfTransformer (required if train=False).
        vocab_size: Size of the visual vocabulary (default 100).

    Returns:
        If train: dict with trained models and accuracy.
        If test: accuracy or predictions.
    """
    sift = SIFT()
    sift_features = []
    y_features = []

    for idx in tqdm(range(X.shape[0]), desc="Extracting SIFT features"):
        try:
            img = X[idx].reshape(32, 32, 3)
            img = color.rgb2gray(img)
            sift.detect_and_extract(img)
            if sift.descriptors is not None and len(sift.descriptors) > 0:
                sift_features.append(sift.descriptors)
                y_features.append(y[idx])
        except Exception as e:
            # print(f"Error processing image {idx}: {e}")
            pass

    if len(sift_features) == 0:
        print("No SIFT features extracted from any image.")
        return None

    if train:
        # Training phase
        stack_sift_features = np.concatenate(sift_features)
        print(f"Total SIFT descriptors: {stack_sift_features.shape[0]}")
        kmeans = KMeans(n_clusters=vocab_size, random_state=23)
        kmeans.fit(stack_sift_features)
        print(f"Created vocabulary with {vocab_size} clusters")

    # Encoding features
    image_histograms = []
    for feature in tqdm(sift_features, desc="Encoding SIFT features"):
        clusters = kmeans.predict(feature)
        histogram, _ = np.histogram(
            clusters, bins=vocab_size, range=(0, vocab_size))
        image_histograms.append(histogram)
    image_histograms_np = np.array(image_histograms)

    if train:
        tfidf = TfidfTransformer()
        tfidf.fit(image_histograms_np)
        image_histograms_tfidf = tfidf.transform(image_histograms_np)
        svm = LinearSVC(random_state=23)
        svm.fit(image_histograms_tfidf, y_features)
        accuracy = svm.score(image_histograms_tfidf, y_features)
        print(f"SVM accuracy on training data: {accuracy * 100:.2f}%")
        return image_histograms_np, y_features, svm, kmeans, tfidf
    else:
        if svm is None or kmeans is None or tfidf is None:
            raise ValueError(
                "svm, kmeans, and tfidf must be provided for testing.")
        image_histograms_tfidf = tfidf.transform(image_histograms_np)
        accuracy = svm.score(image_histograms_tfidf, y_features)
        print(f"SVM accuracy on test data: {accuracy * 100:.2f}%")
        return image_histograms_np, y_features
