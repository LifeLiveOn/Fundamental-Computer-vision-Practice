import cv2
import numpy as np
import urllib.request
import argparse

"""
This script converts an image from RGB to HSV, applies hyperparameters to adjust hue, saturation, and brightness,
and then converts it back to RGB for display. The hyperparameters can be adjusted via command line arguments.

I use opencv so I have to convert the image from BGR to RGB since opencv reads images in BGR format.
pixel values before going to cv has to be in uint8 format, so I convert the image to uint8 after processing.


"""

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYVx6CB56pxO8gwlzLLOkV8fPN0jfF3T_98w&s"
res = urllib.request.urlopen(url)
# Read the image from the URL, this is just a 1D byte array
image = np.asarray(bytearray(res.read()), dtype="uint8")
# Decode the image to a format of Width x Height x Channels
img = cv2.imdecode(image, cv2.IMREAD_COLOR)
height, width, _ = img.shape


def RGBtoHSV(img):
    """
    Fully vectorized RGB to HSV conversion using NumPy.
    """
    # Convert to RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    R, G, B = np.split(rgb, 3, axis=2) #split the channels by 3 targeting the last dimension
    R,G,B = R.squeeze(), G.squeeze(), B.squeeze()  # Remove singleton d

    V = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    C = V - min_c

    # Hue
    H = np.zeros_like(V)
    mask_r = np.where((V == R) & (C != 0))  # R is max and C is not zero
    mask_g = np.where((V == G) & (C != 0))
    mask_b = np.where((V == B) & (C != 0))

    # for R max → base sector 0°
    H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / C[mask_r]))
    H[mask_g] = (60 * ((B[mask_g] - R[mask_g]) / C[mask_g]) + 120)
    H[mask_b] = (60 * ((R[mask_b] - G[mask_b]) / C[mask_b]) + 240)

    # Saturation
    S = np.zeros_like(V)
    mask_v = V != 0
    S[mask_v] = C[mask_v] / V[mask_v]

    return H, S, V


# convert HSV back to RGB


def HSVtoRGB(H, S, V):
    """
    Convert HSV channels to RGB format.
    Parameters:
        H (numpy.ndarray): Hue channel in degrees.
        S (numpy.ndarray): Saturation channel in range [0, 1].
        V (numpy.ndarray): Value channel in range [0, 1]."""
    C = V * S  # Chroma shape is same as V
    H_ = H / 60  # Normalize H to [0, 6]
    X = C * (1 - np.abs(H_ % 2 - 1))
    m = V - C

    R, G, B = np.zeros_like(V), np.zeros_like(V), np.zeros_like(V)

    # Hue region 0°–60°: R = C, G = X, B = 0
    mask0 = (H_ >= 0) & (H_ < 1)
    R[mask0], G[mask0], B[mask0] = C[mask0], X[mask0], 0

    # 60°–120°: R = X, G = C, B = 0
    mask1 = (H_ >= 1) & (H_ < 2)
    R[mask1], G[mask1], B[mask1] = X[mask1], C[mask1], 0

    # 120°–180°: R = 0, G = C, B = X
    mask2 = (H_ >= 2) & (H_ < 3)
    R[mask2], G[mask2], B[mask2] = 0, C[mask2], X[mask2]

    # 180°–240°: R = 0, G = X, B = C
    mask3 = (H_ >= 3) & (H_ < 4)
    R[mask3], G[mask3], B[mask3] = 0, X[mask3], C[mask3]

    # 240°–300°: R = X, G = 0, B = C
    mask4 = (H_ >= 4) & (H_ < 5)
    R[mask4], G[mask4], B[mask4] = X[mask4], 0, C[mask4]

    # 300°–360°: R = C, G = 0, B = X
    mask5 = (H_ >= 5) & (H_ < 6)
    R[mask5], G[mask5], B[mask5] = C[mask5], 0, X[mask5]

    # Add m to each channel to match the value
    # axis -1 mean let it stack the last dimension
    # Stack B, G, R into a single array with shape (height, width, 3) and scale to [0, 255]
    return np.stack((B + m, G + m, R + m), axis=2) * 255


# make a decorator to warn value in range
def warn_float(min_val, max_val):
    def checker(value):
        f = float(value)
        if f < min_val or f > max_val:
            raise argparse.ArgumentTypeError(
                f"Value must be in range [{min_val}, {max_val}]")
        return f
    return checker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune HSV hyperparameters for image transformation.")
    # use -- if need to pass arguments from command line, if not use order ex: --h_shift 30 --s_scale 1.2 --v_scale 0.8
    # shift hue by degrees, if reach 360 it will wrap around to 0
    parser.add_argument("h_shift", type=warn_float(-360, 360),
                        default=0, help="Hue shift in degrees (-360, 360).", )
    parser.add_argument("s_scale", type=warn_float(-1, 1), default=1.0,
                        help="Saturation scale (e.g. 0 = no change).",)  # deconstruct the colors to gray if reach 0
    parser.add_argument("v_scale", type=warn_float(-1, 1), default=1.0,
                        help="Brightness (value) scale (e.g. 0 = no change).", )  # value is for lightness
    args = parser.parse_args()

    # Convert to HSV
    H, S, V = RGBtoHSV(img)

    # Apply hyperparameters
    H = (H + args.h_shift) % 360  # Wrap around hue to stay within [0, 360)
    S = np.clip(S + args.s_scale, 0, 1)
    V = np.clip(V + args.v_scale, 0, 1)

    # Convert back to RGB
    img_rgb = HSVtoRGB(H, S, V)

    # Show the results
    cv2.imshow("Original Image", img)
    cv2.imshow("Converted Image", img_rgb.astype(
        np.uint8))  # Convert to uint8 for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()
