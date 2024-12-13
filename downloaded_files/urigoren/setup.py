from distutils.core import setup
with open("requirements.txt", 'r') as f:
    requirements = [l.strip() for l in f.readlines() if l.find("=")>0]

setup(
    name='decorators4DS',
    packages=['decorators4DS'],
    install_requires=requirements,
    version='1.2',
    python_requires='>=3.4',
    description='Useful decorators for data science',
    author='Uri Goren',
    author_email='uri@goren4u.com',
    url='https://github.com/urigoren/decorators4DS',
    download_url='https://github.com/urigoren/decorators4DS/archive/master.zip',
    keywords=['decorator', 'jupyter', 'data science'],
    classifiers=[],
)
