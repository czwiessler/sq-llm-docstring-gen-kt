from setuptools import find_packages, setup

setup(
    name='crypr',
    version='0.0.1',
    packages=find_packages(),
    description='A prediction API for cryptocurrencies.',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='data_science cryptocurrency bitcoin ethereum deep_learning machine_learning prediction',
    author='Daniel C Stevenson',
    author_email='daniel.cortez.stevenson@gmail.com',
    license='MIT',
    install_requires=[
        'connexion==2.2.0',
        'click>=0.7.0',
        'Flask>=1.0.2',
        'Keras>=2.2.2',
        'keras-self-attention==0.36',
        'numpy>=1.16',
        'pandas>=0.23.1',
        'python-dotenv>=0.8.2',
        'PyWavelets==1.0.1',
        'requests>=2.19.1',
        'scikit-learn>=0.19.2',
        'scipy>=1.1.0',
        'tensorflow>=1.8.0',
    ],
    setup_requires=[
        'pytest-runner>=2.0,<3dev',
    ],
    tests_require=[
        'pytest>=4.0.2',
        'coverage>=4.5.2',
        'nbval>=0.9',
        'pytest-xdist>=1.28',
    ],
    entry_points = {
        'console_scripts': [
            'crypr-data=crypr.scripts.make_dataset:main',
            'crypr-features=crypr.scripts.make_features:main',
            'crypr-models=crypr.scripts.make_train_models:main',
        ],
    },
    include_package_data=True,
    zip_safe=False
)
