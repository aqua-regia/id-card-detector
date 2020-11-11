from unittest import TestCase

import cv2

from id_card_extractor.detect_neat_image import get_glare_value


class NeatImageTests(TestCase):
    def setUp(self) -> None:
        self.test_image_filepath = '/Users/syedhassanashraf/Documents/python/extractor/test/image1.png'

    def test_get_glare_value(self):
        """Not RUNNING"""
        image = cv2.imread(self.test_image_filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glare_value = get_glare_value(gray)
        self.assertGreater(glare_value, 0, 'Glare value should be greater than 0')
