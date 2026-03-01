"""Setup configuration for the sentiment-analysis-project package."""

from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = (HERE / "requirements.txt").read_text(encoding="utf-8").splitlines()
# Filter comments and empty lines
REQUIREMENTS = [
    line.strip()
    for line in REQUIREMENTS
    if line.strip() and not line.startswith("#")
]

setup(
    name="sentiment-analysis-project",
    version="0.1.0",
    description="NLP pipeline for sentiment classification of customer reviews",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="NabilDia",
    python_requires=">=3.8",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "word2vec": ["gensim>=4.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
