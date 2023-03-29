import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setuptools.setup(
    name                          = "robel-dclaw-env",
    version                       = "0.0.1",
    author                        = "tomoya-y",
    license                       = 'MIT',
    description                   = "You can receive AWS Service Name.",
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    url                           = "https://github.com/tomoya-yamanokuchi/robel-dclaw-env",
    install_requires              = _requires_from_file('requirements.txt'),
    classifiers                   = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages                      = setuptools.find_packages(),
    python_requires               = ">=3.8",
)

# from setuptools import setup, find_packages
# setup(
#     name='custom_service',
#     version='0.1.0',
#     packages=find_packages("custom_service")
# )
