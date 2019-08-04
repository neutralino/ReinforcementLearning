//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-06-22.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import Python
PythonLibrary.useVersion(2)  // stuck with the System python for now. (i.e. /usr/bin/python -m pip)

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let linalg = Python.import("numpy.linalg")

let n = 5
let nStates = n * n
let gamma = 0.9
let prob = 0.25

// NB: Point(0, 0) is the bottom-left corner in this chapter.
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

// return sebsequent state and reward
func move(_ p: Point, _ action: Action) -> (Point, Int) {
    if p == Point(1, 4) {
        return (Point(1, 0), 10)
    }
    else if p == Point(3, 4){
        return (Point(3, 2), 5)
    }
    switch action {
    case .north:
        if p.y+1 > n-1 {
            return (Point(p.x, p.y), -1)
        }
        else {
            return (Point(p.x, p.y+1), 0)
        }
    case .south:
        if p.y-1 < 0 {
            return (Point(p.x, p.y), -1)
        }
        else {
            return (Point(p.x, p.y-1), 0)
        }
    case .east:
        if p.x+1 > n-1 {
            return (Point(p.x, p.y), -1)
        }
        else {
            return (Point(p.x+1, p.y), 0)
        }
    case .west:
        if p.x-1 < 0 {
            return (Point(p.x, p.y), -1)
        }
        else {
            return (Point(p.x-1, p.y), 0)
        }
    }
}

func getNextStatesAndRewards(_ p: Point) -> (Point, Point, Point, Point, Int, Int, Int, Int) {
    let (northState, northReward) = move(p, Action.north)
    let (southState, southReward) = move(p, Action.south)
    let (eastState, eastReward) = move(p, Action.east)
    let (westState, westReward) = move(p, Action.west)
    return (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward)
}

func generateBellmanEquation(_ p: Point) -> (Double, Point, Point, Point, Point) {
    let (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward) = getNextStatesAndRewards(p)
    let const = prob * Double(northReward + southReward + eastReward + westReward)
    return (const, northState, southState, eastState, westState)
}

func make_value_function_heatmap(valueFunction: PythonObject, title: String, filename: String) {
    let vGrid = np.flipud(np.reshape(valueFunction, Python.tuple([n, n])))

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

func make_figure_3_2() {
    let A = np.zeros([nStates, nStates])
    let b = np.zeros([nStates, 1])

    // Compute the Bellman equation for each state
    for i in 0..<nStates {
        let p = intToGrid(i)
        let (const, northState, southState, eastState, westState) = generateBellmanEquation(p)

        let coef = PythonObject(prob * gamma)

        // fill row in matrix
        A[i, gridToInt(northState)] += coef
        A[i, gridToInt(southState)] += coef
        A[i, gridToInt(eastState)] += coef
        A[i, gridToInt(westState)] += coef

        // move p to RHS
        A[i, gridToInt(p)] -= 1

        // move const to LHS
        b[i] = PythonObject(-const)
    }

    // solve for state-value function
    let v = linalg.solve(A, b)

    make_value_function_heatmap(valueFunction: v, title: "Gridworld state-value function", filename: "Fig_3.2.png")
}

func make_figure_3_5() {
    //solve via value iteration (see Section 4.4)
    let valueFunction = np.zeros([nStates, 1])

    var delta = 0.0
    let minDelta = 1e-5
    var count = 0
    repeat {
        delta = 0.0
        count += 1
        for i in 0..<nStates {
            let p = intToGrid(i)

            let (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward) = getNextStatesAndRewards(p)

            // determine the best value by looping through all actions
            let expectedNorthActionValue = Double(northReward) + gamma * Double(valueFunction[gridToInt(northState)])!
            let expectedSouthActionValue = Double(southReward) + gamma * Double(valueFunction[gridToInt(southState)])!
            let expectedWestActionValue  = Double(westReward)  + gamma * Double(valueFunction[gridToInt(westState)])!
            let expectedEastActionValue  = Double(eastReward)  + gamma * Double(valueFunction[gridToInt(eastState)])!
            let nextV = max(expectedNorthActionValue, expectedSouthActionValue, expectedWestActionValue, expectedEastActionValue)

            delta = max(delta, abs(Double(valueFunction[i])! - nextV))
            valueFunction[i] = PythonObject(nextV)
        }
        print("Finished \(count) iterations, delta = \(delta)")
    } while delta > minDelta

    make_value_function_heatmap(valueFunction: valueFunction, title: "Gridworld optimal state-value function", filename: "Fig_3.5.png")
}

make_figure_3_2()
make_figure_3_5()
