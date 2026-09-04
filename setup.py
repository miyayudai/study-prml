from setuptools import setup, find_packages

setup(
    name="prml",
    version="1.0.0",
    description="Comprehensive Python Implementation for Pattern Recognition and Machine Learning",
    author="miyayudai",
    packages=find_packages(include=["common*", "prml*"]),
    package_data={"common": ["faithful.csv"]},
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.9",
)
