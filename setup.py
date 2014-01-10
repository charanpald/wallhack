
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
    install_requires=['numpy>=1.5.0', 'scipy>=0.7.1', "scikit-learn>=0.13"],
    long_description="A collection of experiments with machine learning algorithms",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License"
    ],
)
