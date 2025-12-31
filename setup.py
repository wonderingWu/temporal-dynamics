#!/usr/bin/env python3
"""
Setup script for temporal-dynamics project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="temporal-dynamics",
    version="1.0.0",
    author="[Author Name]",
    author_email="[email@institution.edu]",
    description="Information-theoretic analysis of temporal dynamics in complex systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/temporal-dynamics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "temporal-dynamics=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.csv", "*.pdf", "*.png"],
    },
)