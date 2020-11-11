import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="id_card_extractor",
    version="1.1.8",
    description="Crop id card from an image using OpenCV",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/aqua-regia/id-card-detector",
    author="Syed Hassan Ashraf",
    author_email="hassanashraf8888@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=["id_card_extractor"],
    include_package_data=True,
    install_requires=[],
    entry_points={
    },
)