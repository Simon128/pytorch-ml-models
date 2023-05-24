from setuptools import setup, find_packages

setup(
    name='pytorch-ml-models',
    version='0.0.1',
    packages=find_packages(include=['models', 'models.*']),
    install_requires=[
    ],
)


