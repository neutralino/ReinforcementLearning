#!/usr/bin/env bash

if swiftc -O -sdk `xcrun --show-sdk-path` main.swift GridWorld.swift PoissonDistribution.swift CarRental.swift GamblersProblem.swift -o chapter_4 ; then
    ./chapter_4
else
    echo "Compilation failed"
fi
