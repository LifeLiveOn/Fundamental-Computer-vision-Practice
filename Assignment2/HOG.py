import matplotlib.pyplot as plt
import numpy as np
from skimage import data, exposure
from skimage.feature import hog, match_descriptors

fd, hog_image = hog(data.astronaut(), pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(data.astronaut(), cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('HOG features (rescaled)')
plt.show()

# this image is describe by number of <result of this print> vectors
print("HOG feature descriptor shape:", fd.shape)


def extract_hog_features(image):
    """
    Extract HOG features from the input image.

    Args:
        image: Input image (numpy array).

    Returns:
        hog_features: HOG feature descriptor.
        hog_image: Visual representation of HOG features.
    """
    hog_features, hog_image = hog(image, pixels_per_cell=(16, 16),
                                  cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return hog_features, hog_image
