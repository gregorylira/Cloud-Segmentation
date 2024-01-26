from setuptools import setup, find_packages

setup(
    name='fpnmodel',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='gregorylira',
    description='Dataset class for the Cloud Segmentation project',
    py_modules=['fpn_model', 'fpnHead', 'UpsampleModule', 'SemibasicBlock'],
    url='',
)