#!/usr/bin/env bash

if swiftc -O -sdk `xcrun --show-sdk-path` main.swift BlackJack.swift InfiniteVariance.swift -o chapter_5 ; then
    ./chapter_5
else
    echo "Compilation failed"
fi
