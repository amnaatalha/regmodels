from setuptools import setup, find_packages

setup(
    name="regmodels",
    version="0.1.0",
    author="Amna Talha",
    author_email="your_email@example.com",
    description="A regression model comparison and visualization library with statistical analysis and theme support.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/regmodels",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "IPython"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
