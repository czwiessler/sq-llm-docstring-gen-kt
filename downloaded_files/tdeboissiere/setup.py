import setuptools

setuptools.setup(
    name="supernnova",
    version="0.0.2",
    author="Anais Moller and Thibault de Boissiere",
    author_email = "anais.moller@anu.edu.au",
    description = "framework for Bayesian, Neural Network based supernova light-curve classification",
    url="https://github.com/tdeboissiere/SuperNNova",
    packages=setuptools.find_packages(),
    classifiers=[
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   ],
    extras_require={
        "docs": [
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
            "sphinx-autobuild",
            "sphinxcontrib-napoleon",
        ]
    },
)
