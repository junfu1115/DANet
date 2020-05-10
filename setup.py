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

cwd = os.path.dirname(os.path.abspath(__file__))

version = '1.2.1'
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
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

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'portalocker',
    'torch>=1.4.0',
    'torchvision>=0.5.0',
    'Pillow',
    'scipy',
    'requests',
]

if __name__ == '__main__':
    create_version_file()
    setup(
        name="torch-encoding",
        version=version,
        author="Hang Zhang",
        author_email="zhanghang0704@gmail.com",
        url="https://github.com/zhanghang1989/PyTorch-Encoding",
        description="PyTorch Encoding Package",
        long_description=open('README.md').read(),
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
    )
