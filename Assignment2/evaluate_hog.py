from skimage.feature import hog
from skimage import color
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.svm import LinearSVC, SVC


def extract_hog_features(X, y_features, train=True, svm=None, kmeans=None, tfidf=None, vocab_size=200):
    """
    Extract HOG features from images and train or test using SVM
    Detect keypoints and then extract local patch when we slide a window of (8,8) , each cell is (4,4) pixels, each block is (2,2) cells
    we create a feature vectors per keypoint then we use k-means to cluster all those features into fixed number of clusters (vocab_size)
    and then we use the histogram of visual words to represent the image, we can use tf-idf to normalize the histogram of visual words
    and then we use SVM to classify the images based on the histogram of visual words
    Args:
        X: Images (numpy array).
        y_features: Labels for the images (numpy array).
        train: If True, train the model. If False, test using provided models.
        svm: Trained SVM model (required if train=False).
        kmeans: Trained KMeans model (required if train=False).
        tfidf: Trained TfidfTransformer (required if train=False).
        vocab_size: Size of the visual vocabulary (default 200).
    """
    hog_features = []
    for idx in tqdm(range(X.shape[0]), desc="Extracting HOG features"):
        try:
            img = X[idx].reshape(32, 32, 3)
            img = color.rgb2gray(img)
            hog_feature = hog(img, pixels_per_cell=(
                8, 8), cells_per_block=(2, 2), visualize=False)
            hog_features.append(hog_feature)
        except Exception as e:
            # print(f"Error processing image {idx}: {e}")
            pass
    if len(hog_features) == 0:
        print("No HOG features extracted")
        return None
    hog_features_np = np.array(hog_features)
    print(f"Total HOG features extracted: {hog_features_np.shape[0]}")
    clf = SVC(random_state=23, kernel='rbf')
    # test before passing to cluster, histogram etc
    y_features = np.array(y_features)
    clf.fit(hog_features_np, y_features)
    # score
    accuracy = clf.score(hog_features_np, y_features)
    print(
        f"SVM accuracy on training data (HOG) before going through bovw: {accuracy * 100:.2f}%")
    if train:
        kmeans = KMeans(n_clusters=vocab_size, random_state=23)
        kmeans.fit(hog_features_np)
    # encoding the features which is the histogram of visual words, we first extract the features from the images then we use k means to cluster the features into vocab_size clusters so then later on we can use a fixed length vector to represent the image to train the SVM

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
        # tfidf is used to normalize the histogram of visual words, it helps to reduce the impact of common words and highlight the important ones
        image_histograms_tfidf = tfidf.transform(image_histograms_np)
        svm = SVC(random_state=23, kernel='rbf')
        svm.fit(image_histograms_tfidf, y_features)
        accuracy = svm.score(image_histograms_tfidf, y_features)
        print(f"SVM accuracy on training data (HOG): {accuracy * 100:.2f}%")
        return image_histograms_np, y_features, svm, kmeans, tfidf
    else:
        if svm is None or kmeans is None or tfidf is None:
            raise ValueError(
                "svm, kmeans, and tfidf must be provided for testing.")
        image_histograms_tfidf = tfidf.transform(image_histograms_np)
        accuracy = svm.score(image_histograms_tfidf, y_features)
        print(f"SVM accuracy on test data (HOG): {accuracy * 100:.2f}%")
        return image_histograms_np, y_features
