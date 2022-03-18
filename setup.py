import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="spindleTracker",
    version="1.0",
    author="lizhogn",
    python_requires=">=3.6",
    long_description=long_description,
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_packages(),
)
