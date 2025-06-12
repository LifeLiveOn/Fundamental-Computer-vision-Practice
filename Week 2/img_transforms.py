import cv2
import numpy as np
import urllib.request
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYVx6CB56pxO8gwlzLLOkV8fPN0jfF3T_98w&s"
res = urllib.request.urlopen(url)
# Read the image from the URL, this is just a 1D byte array
image = np.asarray(bytearray(res.read()), dtype="uint8")
# Decode the image to a format of Width x Height x Channels
img = cv2.imdecode(image, cv2.IMREAD_COLOR)


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
    assert isinstance(crop_size, tuple) and len(crop_size) == 2, "crop_size must be a tuple of (height, width)"
    assert min(crop_size) > 0, "Crop size must be positive integers."

    if h < crop_h or w < crop_w:
        raise ValueError("Crop size must be smaller than the image size.")
    
    center_y = h // 2
    center_x = w // 2

    #Random off set within [-crop_h//2, crop_h//2] and [-crop_w//2, crop_w//2]
    offset_y = np.random.randint(-crop_h // 2, crop_h // 2 + 1)
    offset_x = np.random.randint(-crop_w // 2, crop_w // 2 + 1)

    # Clip is use to ensure values are from 0 to h-crop_h and 0 to w-crop_w
    start_y = np.clip(a = center_y + offset_y - crop_h // 2, a_min=0,a_max= h - crop_h)
    start_x = np.clip(a = center_x + offset_x - crop_w // 2, a_min=0,a_max= w - crop_w) 
    
    crop_img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    # print(crop_img.shape) 
    return crop_img

# cv2.imshow("Original Image", img.astype(np.uint8))
# cv2.imshow("Cropped Image", random_crop(img=img, crop_size=(120, 120)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def extract_patch(img, num_patches):
    
