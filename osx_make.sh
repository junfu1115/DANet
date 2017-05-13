#!/usr/bin/env bash

cd encoding/
mkdir -p build && cd build

cmake ..
make install

cd ../..
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
