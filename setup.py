from setuptools import setup, find_packages

setup(
    name        = 'service',
    version     = '0.1.0',
    description = 'service domain',
    packages    = find_packages(where='service'),
    package_dir = {'': 'service'},
)