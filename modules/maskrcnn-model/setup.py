from setuptools import setup, find_packages

setup(
    name='masrcnnmodel',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='gregorylira',
    description='Dataset class for the Cloud Segmentation project',
    py_modules=['maskrcnn_model', 'maskrcnn_lightning'],
    url='',
)