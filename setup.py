from setuptools import find_packages, setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='METNetwork',
    version='0.1',
    description='Training and evaluating the METNetwork',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mattcleigh/METNetwork.git',
    author='Matthew Leigh',
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',
        'tqdm',
        'pandas',
    ],
    dependency_links=[],
)
