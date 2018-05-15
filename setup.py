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

version = '0.4.0'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
        cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
except Exception:
    pass

try:
    import pypandoc
    readme = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    readme = open('README.md').read()

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'torch>=0.3.1',
    'cffi>=1.0.0',
]

setup(
    name="encoding",
    version=version,
    author="Hang Zhang",
    author_email="zhanghang0704@gmail.com",
    url="https://github.com/zhanghang1989/PyTorch-Encoding",
    description="PyTorch Encoding Package",
    long_description=readme,
    license='MIT',
    install_requires=requirements,
    packages=find_packages(exclude=["tests", "experiments"]),
    package_data={ 'encoding': [
        'lib/*.so*', 'lib/*.dylib*',
        '_ext/encoding_lib/*.so', '_ext/encoding_lib/*.dylib',
        'kernel/*.h', 'kernel/generic/*h',
        'src/*.h',
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
