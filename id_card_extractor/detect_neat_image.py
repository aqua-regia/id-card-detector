# Import packages

import cv2


def get_glare_value(gray):
    """
    :param gray: cv2.imread(image_path) grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    :return: numrical value between 0-256 which tells the glare value
    """
    blur = cv2.blur(gray, (3, 3))  # With kernel size depending upon image size
    mean_blur = cv2.mean(blur)
    return mean_blur[0]


def get_blur_value(gray):
    """
    :param gray: cv2.imread(image_path) grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    :return: numrical value between 0-256 which tells the blur
    """

    # You simply take a single channel of an image (presumably grayscale) and convolve it with the following 3 x 3 kernel and then take the variance
    # and use a threshold
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


if __name__ == '__main__':
    test_image_filepath = '/Users/syedhassanashraf/Documents/python/extractor/test/image1.png'
    image = cv2.imread(test_image_filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glare_value = get_glare_value(gray)
    blur_value = get_blur_value(gray)
    print(f"glare and blur values are:- {glare_value} {blur_value}")
