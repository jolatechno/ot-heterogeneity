from setuptools import setup, find_packages

setup(
    name='oterogeneity',
    version='0.1.5',
    description='Python library to compute heterogeneity indexes based on optimal transport.',
    long_description='Python library to compute heterogeneity indexes based on optimal transport.',
    readme="README.md",
    author='Joseph TOUZET',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'pot']
)
