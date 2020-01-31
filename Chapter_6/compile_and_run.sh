#!/usr/bin/env bash

if swiftc -O -sdk `xcrun --show-sdk-path` main.swift RandomWalk.swift GridWorld.swift -o chapter_6 ; then
    ./chapter_6
else
    echo "Compilation failed"
fi
