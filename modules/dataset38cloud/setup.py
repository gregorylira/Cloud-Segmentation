from setuptools import setup, find_packages

setup(
    name='dataset38cloud',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='gregorylira',
    description='Dataset class for the Cloud Segmentation project',
    py_modules=['datasetclass_38', 'datamodule_38', 'transforms_class'],
    url='',
)