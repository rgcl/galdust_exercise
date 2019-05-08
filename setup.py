# -*- coding: utf-8 -*-
# Copyright (C) 2019 Rodrigo González
# Licensed under the MIT licence - see LICENSE.txt
# Author: Rodrigo González-Castillo

from distutils.command.build import build

from setuptools import setup

setup(
    name='galdust_exercise',
    version='0.1',
    packages=['galdust_exercise'],
    entry_points={
        'console_scripts': ['galdust_exercise = galdust_exercise:main']
    },
    install_requires=[
        'numpy', 'matplotlib', 'astropy', 'galdust'
    ],
    dependency_links=[
        'git+ssh://git@github.com/rgcl/galdust.git#egg=galdust',
    ],
    package_data={'galdust_exercise': ['galdust_exercise/data/*']},
    author='Rodrigo González-Castillo',
    author_email='rodrigo.gonzalez@uamail.cl',
    description='Models for the dust emission of galaxies using the data by the astronomer Bruce Draine',
    license='MIT',
    keywords='astrophysics, galaxy, dust emission'
)
