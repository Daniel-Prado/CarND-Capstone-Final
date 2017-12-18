#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    README = open('README.md').read()
except Exception:
    README = ""
VERSION = "0.0.1"
TF_REQUIRE_VERSION = "1.4.0"

requirments = []

from distutils.version import LooseVersion
import pkg_resources
try:
    tf_pkg = pkg_resources.get_distribution("tensorflow")
except pkg_resources.DistributionNotFound:
    try:
        tf_pkg = pkg_resources.get_distribution("tensorflow-gpu")
    except pkg_resources.DistributionNotFound:
        tf_pkg = None

if not tf_pkg:
    requirments.append("tensorflow")
elif LooseVersion(tf_pkg.version) < TF_REQUIRE_VERSION:
    requirments.append(tf_pkg.key + ">=" + TF_REQUIRE_VERSION)


setup(
    name='capstone_traffic_light_classification',
    version=VERSION,
    description='',
    url="http://gitlab.com/carnd-final/final/",
    long_description=README,
    author='Jay Young(yjmade)',
    author_email='carnd@yjmade.net',
    packages=find_packages(),
    install_requires=requirments,
    entry_points={},
)
