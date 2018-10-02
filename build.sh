#!/bin/bash
cd build
#cmake .
#make all
set e

g++ ../src/main.cpp -o screena -lX11 -lXext -Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2 -std=c++17 $(pkg-config --cflags --libs opencv x11 tesseract) &&
    ./screena
