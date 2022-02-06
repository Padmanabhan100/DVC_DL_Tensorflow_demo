import pandas
from setuptools import setup

with open("README.md","r",encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='src',
    version='0.0.1',
    author="padmanabhan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Padmanabhan100/dvc-ml-basics",
    author_email="paddyrolex10@gmail.com",
    packages=['src'],
    python_requires=">=3.6",
    install_requires=[
        "dvc",
        "pandas",
        "scikit-learn",
        "dvc",
        "tensorflow",
        "matplotlib",
        "numpy",
        "pandas",
        "tqdm",
        "PyYAML",
        "boto3",
    ] 
)
