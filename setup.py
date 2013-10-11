import os
from setuptools import setup

setup(
    name = "wallhack",
    version = "0.1",
    author = "Charanpal Dhanjal ",
    author_email = "charanpal@gmail.com",
    description = ("A collection of experiments with machine learning algorithms"),
    license = "GPLv3",
    keywords = "numpy",
    url = "http://packages.python.org/wallhack",
    packages=['wallhack.clusterexp', 'wallhack.egograph', 'wallhack.recommendexp'],
    long_description="A collection of experiments with machine learning algorithms",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License"
    ],
)
