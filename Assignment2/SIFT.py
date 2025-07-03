import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import hog, SIFT, match_descriptors
from skimage import data
import skimage.feature as ft

# sift is invarient to rotation, scale and translation
img1 = rgb2gray(data.astronaut())
img2 = transform.rotate(img1, 45)

descriptor_extractor = SIFT()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def match_features(desc1, desc2):
    """
    Match features between two sets of descriptors using a simple distance metric.
    Args:
        desc1: Descriptors from the first image.
        desc2: Descriptors from the second image.
    Returns:
        matches: a list of tuples (i,j)  where i is the index in desc1 and j is the index in desc2.
    """
    matches = []
    for i, d1 in enumerate(desc1):
        similarities = [cosine_similarity(d1, d2) for d2 in desc2]

        j_best = np.argmax(similarities)
        # cross check here
        rev_similarities = [cosine_similarity(
            desc2[j_best], d1) for d1 in desc1]
        i_best = np.argmax(rev_similarities)
        if i_best == i:
            matches.append((i, j_best))
    # print(np.max(similarities), np.mean(similarities))
    return matches


def extract_features_and_plotCompare(img1=None, img2=None, plot=False):
    # Extract features from image 1 and image 2
    descriptor_extractor.detect_and_extract(img1)
    kp1 = descriptor_extractor.keypoints
    desc1 = descriptor_extractor.descriptors
    descriptor_extractor.detect_and_extract(img2)
    kp2 = descriptor_extractor.keypoints
    desc2 = descriptor_extractor.descriptors

    # Match features between the two images
    matches = match_features(desc1, desc2)
    print(f"Number of matches: {len(matches)}")
    # matches = match_descriptors(
    #     desc1, desc2, cross_check=True, metric='cosine')
    # print(f"Number of matches using skimage: {len(matches)}")
    # Plot the matches
    if not plot:
        return kp1, kp2, matches
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))
        plt.gray()

        ft.plot_matched_features(
            img1, img2, keypoints0=kp1, keypoints1=kp2, matches=np.array(matches[::50]), ax=ax)
        ax.set_title("Matches using cosine matching function")
        plt.tight_layout()
        plt.show()


extract_features_and_plotCompare(img1, img2)
