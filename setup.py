from setuptools import setup, find_packages

setup(
    name='oterogeneity',
    version='0.1.3',
    description='Python library to compute heterogeneity indexes based on optimal transport.',
    author='Joseph TOUZET',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'pot']
)
