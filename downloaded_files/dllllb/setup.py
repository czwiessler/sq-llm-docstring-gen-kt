#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ds-pipeline",
    version="0.3.1",
    author="Dmitri Babaev",
    author_email="dmitri.babaev@gmail.com",
    description="Data Science oriented tools, mostly in form of scikit-learn transformers and estimators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dllllb/ds-pipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'scikit-learn',
        'pandas',
        'scipy'
    ]
)
