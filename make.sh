#!/usr/bin/env bash

cd encoding/
mkdir -p build && cd build

cmake ..
make install

cd ..
python setup.py install
