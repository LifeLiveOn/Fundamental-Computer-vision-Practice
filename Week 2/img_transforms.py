import random
from color_space_test import RGBtoHSV, HSVtoRGB
import cv2
import numpy as np
import urllib.request

from numpy.lib import stride_tricks

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYVx6CB56pxO8gwlzLLOkV8fPN0jfF3T_98w&s"
res = urllib.request.urlopen(url)
# Read the image from the URL, this is just a 1D byte array
image = np.asarray(bytearray(res.read()), dtype="uint8")
# Decode the image to a format of Width x Height x Channels
img = cv2.imdecode(image, cv2.IMREAD_COLOR)
# cv2.imshow("Original Image", img.astype(np.uint8))


def random_crop(img, crop_size):
    """
    Randomly crop at center of an image to the specified size.

    Parameters:
        img (numpy.ndarray): Input image.
        crop_size (tuple): Desired crop size (height, width).

    Returns:
        numpy.ndarray: Cropped image.
    """
    h, w, _ = img.shape  # img.shape return (height, width, channels)
    crop_h, crop_w = crop_size
    assert isinstance(crop_size, tuple) and len(
        crop_size) == 2, "crop_size must be a tuple of (height, width)"
    assert min(crop_size) > 0, "Crop size must be positive integers."

    if h < crop_h or w < crop_w:
        raise ValueError("Crop size must be smaller than the image size.")

    center_y = h // 2
    center_x = w // 2

    # Random off set within [-crop_h//2, crop_h//2] and [-crop_w//2, crop_w//2]
    offset_y = np.random.randint(-crop_h // 2, crop_h // 2 + 1)
    offset_x = np.random.randint(-crop_w // 2, crop_w // 2 + 1)

    # Clip is use to ensure values are from 0 to h-crop_h and 0 to w-crop_w
    start_y = np.clip(a=center_y + offset_y - crop_h //
                      2, a_min=0, a_max=h - crop_h)
    start_x = np.clip(a=center_x + offset_x - crop_w //
                      2, a_min=0, a_max=w - crop_w)

    crop_img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    # print(crop_img.shape)
    return crop_img


cv2.imshow("Cropped Image", random_crop(img=img, crop_size=(120, 120)))
cv2.waitKey(0)
cv2.destroyAllWindows()

def extract_patches(img, patch_size):
    H, W, C = img.shape

    # example patch_size = 6, height is 120 so num_patches_h = 120 // 6 = 20 each patch is 6 pixels high
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    # total we should have 20 * 20 = 400 patches of size 6x6

    # compute new shape and strides for extracting patches
    shape = (
        num_patches_h,  # how many patch rows
        num_patches_w,  # how many patch columns
        patch_size,     # rows within each patch
        patch_size,     # cols within each patch
        C               # channels
    )

    # Built-in NumPy tuple that tells how many bytes to jump per axis
    # print(img.strides)
    # img.strides = (W * C * itemsize, C * itemsize, itemsize) for Row, Col, Channel
    # to move to the next patch, our strides will be: think of it as a rule for how to move in the array
    strides = (patch_size * img.strides[0],  # next patch row
               patch_size * img.strides[1],  # next patch col
               img.strides[0],  # next row inside patch
               img.strides[1],  # move col inside patch
               img.strides[2])  # move across chanel
    # Create a sliding window view base on our rules 
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches

patches = extract_patches(img, patch_size=50) #each of size 50x50
cv2.imshow("Extracted Patch", patches[0, 0].astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

def resize_img(img, factor):
    H, W, C = img.shape
    new_H = int(H * factor)
    new_W = int(W * factor)
    img_out = np.zeros((new_H, new_W, C), dtype=img.dtype)
    """Vectorized approach using NumPy"""
    grid_y = np.arange(0, new_H)
    grid_x = np.arange(0, new_W)
    # create a grid of coordinates ex: [grid_y[i, j], grid_x[i, j]] gives the coordinates of pixel (i, j) in the new image
    # grid_y will be each row is 0 0 0 0 then 1 1 1 1 ...
    grid_y, grid_x = np.indices((new_H, new_W))
    # grid_x is like 0 1 2 3 ... then next row is also ....

    """
    So grid_y[i,j] is always i , grid_x[i,j] is always j
    and we want to map these new coordinates back to the original image coordinates
    """

    # map the new grid to original image coordinates
    # -1 so we don't get index at H which is out of bounds
    orig_y = np.clip((grid_y / factor).astype(int), 0, H - 1)
    orig_x = np.clip((grid_x / factor).astype(int), 0, W - 1)
    return img[orig_y, orig_x, :]


cv2.imshow("Resized Image", resize_img(img, factor=0.5).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


def color_jitter(img, hue, saturation, value):
    random_hue = random.uniform(-hue, hue)
    random_saturation = random.uniform(-saturation, saturation)
    random_value = random.uniform(-value, value)
    H, S, V = RGBtoHSV(img)  # Convert to HSV
    # Adjust hue, saturation, and value
    H = (H + random_hue) % 360  # Wrap around hue to stay within [0, 360)
    S = np.clip(S + random_saturation, 0, 1)
    V = np.clip(V + random_value, 0, 1)
    # Convert back to RGB
    img_rgb = HSVtoRGB(H, S, V)  # is actually BGR :))
    return img_rgb.astype(np.uint8)


cv2.imshow("Color Jittered Image", color_jitter(
    img, hue=30, saturation=0.2, value=0.2))
cv2.waitKey(0)
cv2.destroyAllWindows()

