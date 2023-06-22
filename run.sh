#!/bin/bash
clang++ -fno-exceptions -std=c++17 -O3 -mavx2 -ftree-vectorize -mprefer-vector-width=256 -Wall -o floater floater.cc && objdump --disassemble floater >floater.asm && ./floater

