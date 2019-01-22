import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
from distutils.command.build_py import build_py as _build_py
import sys, os.path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amt",
    version="0.0.1",
    author="Martin Zhang",
    author_email="jinye@stanford.edu",
    description="Adaptive Monte Carlo Test",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/martinjzhang/adafdr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    # install_requires=[
    #    'numpy',
    #    'scipy',
    #    'scikit-learn',
    #    'torch==0.3.1',
    #    # 'multiprocessing',
    #    # 'logging',
    #    'matplotlib',
    #    # 'time',
    #    # 'os',
    #    ],
)