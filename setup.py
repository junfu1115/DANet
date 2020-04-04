##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import io
import os
import subprocess

from setuptools import setup, find_packages
import setuptools.command.develop 
import setuptools.command.install 

cwd = os.path.dirname(os.path.abspath(__file__))

version = '1.1.1'
try:
    from datetime import date
    today = date.today()
    day = today.strftime("b%d%m%Y")
    version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'encoding', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is encoding version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

# run test scrip after installation
class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        setuptools.command.install.install.run(self)
        #subprocess.check_call("python tests/unit_test.py".split())

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        setuptools.command.develop.develop.run(self)
        #subprocess.check_call("python tests/unit_test.py".split())

readme = open('README.md').read()

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'torch>=0.5.0',
    'cffi>=1.0.0',
]

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'torch>=0.4.0',
    'Pillow',
    'scipy',
    'requests',
]

setup(
    name="torch-encoding",
    version=version,
    author="Hang Zhang",
    author_email="zhanghang0704@gmail.com",
    url="https://github.com/zhanghang1989/PyTorch-Encoding",
    description="PyTorch Encoding Package",
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',
    install_requires=requirements,
    packages=find_packages(exclude=["tests", "experiments"]),
    package_data={ 'encoding': [
        'LICENSE',
        'lib/cpu/*.h',
        'lib/cpu/*.cpp',
        'lib/gpu/*.h',
        'lib/gpu/*.cpp',
        'lib/gpu/*.cu',
    ]},
    cmdclass={
        'install': install,
        'develop': develop,
    },
)
