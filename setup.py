from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='EcoStock',
    version='1.4',
    author="Antonio Paparo, Giovanni Paparo, Ludovica De Giacomo, Francesco Caldo",
    author_email="antoniopaparo@outlook.com", 
    description='A Python package designed for finance professionals and economists',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tonij10/EcoStock",
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'argparse',
        'uvicorn',
        'gunicorn',
        'pandas',
        'numpy',
        'matplotlib',
        'yfinance',
        'plotly',
        'requests',
        'seaborn',
        'statsmodels',
        'nbformat',
        'scikit-learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
