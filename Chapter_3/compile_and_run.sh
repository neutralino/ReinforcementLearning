#!/usr/bin/env bash

if swiftc -O -sdk `xcrun --show-sdk-path` main.swift GridWorld.swift -o chapter_3 ; then
    ./chapter_3
else
    echo "Compilation failed"
fi
