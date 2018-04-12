#coding=utf-8
"""
imgcv
"""

from setuptools import setup
from setuptools import find_packages

install_requires = [
]

setup(
    name = "imgcv",
    version = "1.0.0",
    description = 'image computer visior',
    author='Hyxbiao',
    author_email="hyxbiao@gmail.com",
    packages = find_packages(),
    entry_points={
        'console_scripts': [
            'imgcv = imgcv.cmdline:imgcv_run',
        ]
    },
    install_requires = install_requires,
    zip_safe = False,
)
