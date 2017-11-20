#!/usr/bin/env bash

mkdir -p encoding/lib && cd encoding/lib
# compile and install
cmake ..
make
