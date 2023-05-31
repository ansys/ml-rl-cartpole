import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="pyansys_gym",
    version="0.0.1",
    author="Jorge E. Gil",
    author_email="jorge.gil@ansys.com",
    description="An OpenAI gym environment based on MAPDL in pyansys",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.pyansys.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
