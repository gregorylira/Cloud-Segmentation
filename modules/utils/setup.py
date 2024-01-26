from setuptools import setup, find_packages

setup(
    name='utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='gregorylira',
    description='Cloud Segmentation project',
    py_modules=['utils'],
    url='',
)