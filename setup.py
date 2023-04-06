from setuptools import setup, find_packages

setup(
    name='pytorch-ml-models',
    version='0.0.1',
    packages=find_packages(include=['model_zoo', 'model_zoo.*']),
    install_requires=[
        'torch==2.0.0'
        'opencv-python==4.7.0.72'
        'torchvision==0.15.1'
        'mmcv==2.0.0'
    ],
)


