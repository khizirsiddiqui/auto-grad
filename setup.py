import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto-grad-khizirsiddiqui", # Replace with your own username
    version="0.0.1",
    author="Khizir Siddiqu",
    author_email="khizirsiddiqui@gmail.com",
    description="Toy Automatic Differentiation Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khizirsiddiqu/auto-grad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)
