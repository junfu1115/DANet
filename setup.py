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

# run test scrip after installation
class install(setuptools.command.install.install):
    def run(self):
        self.create_version_file()
        setuptools.command.install.install.run(self)
        #subprocess.check_call("python tests/unit_test.py".split())
    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'encoding', 'version.py')
        with open(version_path, 'w') as f:
            f.write('"""This is encoding version file."""\n')
            f.write("__version__ = '{}'\n".format(version))

version = '0.3.0'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
        cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
except Exception:
    pass

setup(
    name="encoding",
    version=version,
    description="PyTorch Encoding",
    url="https://github.com/zhanghang1989/PyTorch-Encoding",
    author="Hang Zhang",
    author_email="zhang.hang@rutgers.edu",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    package_data={'encoding': [
        'lib/*.so*', 'lib/*.dylib*',
        'kernel/*.h', 'kernel/generic/*h',
    ]},
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(cwd, "build.py:ffi")
    ],
    cmdclass={
        'install': install,
    },
)
