from math import log2
import numpy as np
import cv2
import urllib.request

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYVx6CB56pxO8gwlzLLOkV8fPN0jfF3T_98w&s"
res = urllib.request.urlopen(url)
# Read the image from the URL, this is just a 1D byte array
image = np.asarray(bytearray(res.read()), dtype="uint8")
# Decode the image to a format of Width x Height x Channels
img = cv2.imdecode(image, cv2.IMREAD_COLOR)


def create_image_pyramid(img, height):
    """
    Create and display an image pyramid with a given height.

    Parameters:
        img (np.ndarray): Input image.
        height (int): Number of pyramid levels including original.
    """
    current_img = img.copy()
    cv2.imshow("img", current_img)  # original image

    for i in range(1, height):
        current_img = cv2.resize(
            current_img,
            (current_img.shape[1] // 2, current_img.shape[0] // 2),
            interpolation=cv2.INTER_LINEAR
        )
        scale = 2 ** i  # e.g., 2x, 4x, 8x
        cv2.imshow(f"img_{scale}x", current_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


create_image_pyramid(img, height=4)
