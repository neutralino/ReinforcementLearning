//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-08-04.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import Python
PythonLibrary.useVersion(2)  // stuck with the System python for now. (i.e. /usr/bin/python -m pip)

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let linalg = Python.import("numpy.linalg")

let n = 4
let nStates = n * n
let gamma = 1.0
let prob = 0.25

// NB: Point(0, 0) is the top-left corner in this chapter.
struct Point: Equatable {
    var x: Int
    var y: Int
    init(_ x: Int, _ y: Int) {
        self.x = x
        self.y = y
    }
}

enum Action {
    case north
    case south
    case east
    case west
}

func intToGrid(_ i: Int) -> Point {
    precondition(i < nStates, "Index must map to valid grid state.")
    return Point(i % n, i / n)
}

func gridToInt(_ p: Point) -> Int {
    return n * p.y + p.x
}

// return subsequent state and reward
func move(_ p: Point, _ action: Action) -> (Point, Int) {
    if p == Point(0, 0) {
        return (Point(0, 0), 0)
    }
    else if p == Point(n-1, n-1) {
        return (Point(0, 0), 0)
    }
    let reward = -1
    switch action {
    case .north:
        return (Point(p.x, min(p.y+1, n-1)), reward)
    case .south:
        return (Point(p.x, max(p.y-1, 0)), reward)
    case .east:
        return (Point(min(p.x+1, n-1), p.y), reward)
    case .west:
        return (Point(max(p.x-1, 0), p.y), reward)
    }
}

func getNextStatesAndRewards(_ p: Point) -> (Point, Point, Point, Point, Int, Int, Int, Int) {
    let (northState, northReward) = move(p, Action.north)
    let (southState, southReward) = move(p, Action.south)
    let (eastState, eastReward) = move(p, Action.east)
    let (westState, westReward) = move(p, Action.west)
    return (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward)
}

func make_value_function_heatmap(valueFunction: PythonObject, title: String, filename: String) {
    let vGrid = np.reshape(valueFunction, Python.tuple([n, n]))
    let fig = plt.figure(figsize: [6.4, 4.8])
    let ax = fig.gca()

    ax.imshow(vGrid, cmap: "YlGn", interpolation: "none")
    for i in 0..<n {
        for j in 0..<n {
            ax.text(j, i, np.around(vGrid[i, j], 1), ha: "center", va: "center", color: "b")
        }
    }
    ax.set_title(title)
    ax.tick_params(axis: "x",labelbottom: "off")
    ax.tick_params(axis: "y",labelleft: "off")
    plt.savefig(filename)
}

func make_figure_4_1() {
    // perform iterative policy evaluation for the random policy
    let v = np.zeros([nStates, 1])
    var delta = 0.0
    let minDelta = 1e-5
    var count = 0
    repeat {
        delta = 0.0
        count += 1
        for i in 0..<nStates {
            let p = intToGrid(i)
            let (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward) = getNextStatesAndRewards(p)

            // determine the next iteration of the state's value
            let expectedNorthActionValue = Double(northReward) + gamma * Double(v[gridToInt(northState)])!
            let expectedSouthActionValue = Double(southReward) + gamma * Double(v[gridToInt(southState)])!
            let expectedWestActionValue  = Double(westReward)  + gamma * Double(v[gridToInt(westState)])!
            let expectedEastActionValue  = Double(eastReward)  + gamma * Double(v[gridToInt(eastState)])!
            let nextV = prob * (expectedNorthActionValue + expectedSouthActionValue + expectedWestActionValue + expectedEastActionValue)
            delta = max(delta, abs(Double(v[i])! - nextV))
            v[i] = PythonObject(nextV)
        }
        print("Finished \(count) iterations, delta = \(delta)")
    } while delta > minDelta
    make_value_function_heatmap(valueFunction: v, title: "Small gridworld state-value function", filename: "Fig_4.1.png")
}

make_figure_4_1()
