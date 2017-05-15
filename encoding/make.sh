#!/usr/bin/env bash

mkdir -p encoding/build && cd encoding/build
# compile and install
cmake ..
make install
cd ..
