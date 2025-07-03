from skimage.feature import hog
from skimage import color
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.svm import LinearSVC


def extract_hog_features(X, y_features, train=True, svm=None, kmeans=None, tfidf=None, vocab_size=100):
    hog_features = []
    for idx in tqdm(range(X.shape[0]), desc="Extracting HOG features"):
        try:
            img = X[idx].reshape(32, 32, 3)
            img = color.rgb2gray(img)
            hog_feature = hog(img, pixels_per_cell=(
                4, 4), cells_per_block=(2, 2), visualize=False)
            hog_features.append(hog_feature)
        except Exception as e:
            # print(f"Error processing image {idx}: {e}")
            pass
    if len(hog_features) == 0:
        print("No HOG features extracted from any image.")
        return None
    hog_features_np = np.array(hog_features)
    if train:
        kmeans = KMeans(n_clusters=vocab_size, random_state=23)
        kmeans.fit(hog_features_np)
    # encoding the features
    image_histograms = []
    for feature in tqdm(hog_features, desc="Encoding HOG features"):
        clusters = kmeans.predict(feature.reshape(1, -1))
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
