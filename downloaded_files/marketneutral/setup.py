from setuptools import setup, find_packages

setup(
    name='alphatools',
    version='0.15',
    description='Quant finance resarch tools',
    author='Jonathan Larkin',
    author_email='jonathan.r.larkin@gmail.com',
    url = "https://github.com/marketneutral/alphatools",
    download_url = "https://github.com/marketneutral/alphatools/archive/0.11.tar.gz",
    packages=find_packages(),
    python_requires='>=3.5.*',
    install_requires=[
        'zipline<=1.3',
        'alphalens',
        'ipykernel',
        'lark-parser',
        'autopep8',
        'bottleneck',
        'tqdm',
        'pydot'
    ],
    entry_points={
        'console_scripts': [
            'alphatools = alphatools.__main__:main',
        ]
    }
)
