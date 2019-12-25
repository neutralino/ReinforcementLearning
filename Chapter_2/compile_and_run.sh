#!/usr/bin/env bash

if swiftc -O -sdk `xcrun --show-sdk-path` main.swift TenArmedTestbed.swift GaussianDistribution.swift DiscreteDistribution.swift -o chapter_2 ; then
    ./chapter_2
else
    echo "Compilation failed"
fi
