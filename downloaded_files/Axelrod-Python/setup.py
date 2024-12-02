from setuptools import setup, find_packages
import unittest
import doctest
import os

# Read in the version number
exec(open('src/axelrod_dojo/version.py', 'r').read())

# Read in the requirements.txt file
with open('requirements.txt') as f:
    requirements = []
    for library in f.read().splitlines():
        if "docopt" not in library:  # Skip: used only for command line scripts
            requirements.append(library)

setup(
    name='axelrod_dojo',
    version=__version__,
    install_requires=requirements,
    author='Marc Harper; Vince Knight; Martin Jones; Georgios Koutsovoulos',
    packages=find_packages('src'),
    package_dir={"": "src"},
    url='',
    license='The MIT License (MIT)',
    description='A library to train strategies for the Iterated Prisoners Dilemma',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        ],
    python_requires='>=3.0',
)
