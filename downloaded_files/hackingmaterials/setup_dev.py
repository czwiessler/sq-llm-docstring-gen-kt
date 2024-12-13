"""Bench-dev (HT-benchmarking) package. If you're looking to install
automatminer (the regular package), just use setup.py."""

from setuptools import setup, find_packages
import os

from automatminer_dev import __version__

module_dir = os.path.dirname(os.path.abspath(__file__))
reqs_raw = open(os.path.join(module_dir, "requirements_dev.txt")).read()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

if __name__ == "__main__":
    setup(
        name='automatminer_dev',
        version=__version__,
        description='benchmarking infrastructure for automatminer',
        long_description="",
        url='https://github.com/hackingmaterials/automatminer',
        author=['Alex Dunn'],
        author_email='ardunn@lbl.gov',
        license='modified BSD',
        packages=find_packages(include="./automatminer_dev"),
        package_data={},
        zip_safe=False,
        install_requires=reqs_list,
        extras_require={},
        classifiers=[])
