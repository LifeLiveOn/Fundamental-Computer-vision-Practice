import cv2
import numpy as np
import urllib.request
import argparse
from color_space_test import warn_float
from PIL import Image
"""
This script converts an image from RGB to HSV, applies hyperparameters to adjust hue, saturation, and brightness NAIVELY
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


def naive_RGBtoHSV(img):
    H_list, S_list, V_list = [], [], []
    for i in range(len(img)):
        for j in range(len(img[0])):
            B, G, R = img[i][j]  # open CV default to BGR

            # Normalize it to [0, 1]
            r, g, b = R / 255.0, G / 255.0, B / 255.0

            # Compute max, min
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            diff = max_c - min_c

            # Hue calculation FORMULA
            if diff == 0:
                H = 0
            elif max_c == r:
                H = 60 * (((g - b) / diff) % 6)
            elif max_c == g:
                H = 60 * (((b - r) / diff) + 2)
            else:  # max_c == b
                H = 60 * (((r - g) / diff) + 4)

            # Saturation
            S = 0 if max_c == 0 else diff / max_c

            # Value
            V = max_c

            H_list.append(H)
            S_list.append(S)
            V_list.append(V)

    return H_list, S_list, V_list


def naive_HSVtoRGB(H, S, V, height, width):
    R, G, B = [], [], []
    for i in range(len(H)):
        h = H[i]
        s = S[i]
        v = V[i]

        C = v * s
        H_ = h / 60
        X = C * (1 - abs(H_ % 2 - 1))
        m = v - C

        if 0 <= H_ < 1:
            r, g, b = C, X, 0
        elif 1 <= H_ < 2:
            r, g, b = X, C, 0
        elif 2 <= H_ < 3:
            r, g, b = 0, C, X
        elif 3 <= H_ < 4:
            r, g, b = 0, X, C
        elif 4 <= H_ < 5:
            r, g, b = X, 0, C
        elif 5 <= H_ < 6:
            r, g, b = C, 0, X
        else:
            r, g, b = 0, 0, 0

        R.append(int(round((r + m) * 255)))
        G.append(int(round((g + m) * 255)))
        B.append(int(round((b + m) * 255)))

    pixels = [(r, g, b) for r, g, b in zip(R, G, B)]
    img = Image.new("RGB", (width, height))
    img.putdata(pixels)
    img.show()
    return img


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
    H, S, V = naive_RGBtoHSV(img)
    # Apply hyperparameters
    H = [(h + args.h_shift) % 360 for h in H]
    S = [min(max(s + args.s_scale, 0), 1) for s in S]
    V = [min(max(v + args.v_scale, 0), 1) for v in V]

    # Convert back to RGB
    img_rgb = naive_HSVtoRGB(H, S, V, height, width)

    # Show the results
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
