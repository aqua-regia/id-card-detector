# Import packages
import os
import sys
from os import listdir
from os.path import isfile, join

import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
CWD_PATH = os.getcwd()


def get_glare_value(image):
    """
    :param image: cv2.imread(image_path)
    :return: numrical value between 0-256 which tells the glare value
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", gray)
    blur = cv2.blur(gray, (3, 3))  # With kernel size depending upon image size
    mean_blur = cv2.mean(blur)
    return mean_blur[0]


def get_blur_value(image):
    """
    :param image: cv2.imread(image_path)
    :return: numrical value between 0-256 which tells the blur
    """

    # You simply take a single channel of an image (presumably grayscale) and convolve it with the following 3 x 3 kernel and then take the variance
    # and use a threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", gray)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


# Path to image
DIRECTORY_NAME = 'test_images/detected'
DIRECTORY_PATH = os.path.join(CWD_PATH, DIRECTORY_NAME)
onlyfiles = [f for f in listdir(DIRECTORY_PATH) if isfile(join(DIRECTORY_PATH, f))]

for filename in onlyfiles:
    image_path = DIRECTORY_PATH + "/" + filename
    image = cv2.imread(image_path)
    glare = get_glare_value(image)
    blur = get_blur_value(image)
    print(f'glare and blur values are:- {glare} {blur}')
    cv2.waitKey(0)
