# pylint: disable=missing-docstring
# This file is part of mapo.

# mapo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mapo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mapo. If not, see <http://www.gnu.org/licenses/>.


import os
from setuptools import setup, find_packages

from mapo.version import __version__


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, "r")
    return file.read()


setup(
    name="mapo",
    version=__version__,
    author="Thiago P. Bueno, Ângelo G. Lovatto",
    author_email="thiago.pbueno@gmail.com, angelolovatto@gmail.com",
    description="MAPO: Model-Aware Policy Optimization",
    long_description=read("README.md"),
    license="GNU General Public License v3.0",
    keywords=["reinforcement-learning", "model-based", "rllib", "tensorflow"],
    url="",
    packages=find_packages(),
    scripts=[],
    install_requires=[
        "gym",
        "ray[rllib]==0.7.2",
        "tensorflow<2.0.0",
        "pandas",
        "requests",
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
