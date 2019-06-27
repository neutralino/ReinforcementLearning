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

func generateEquation(_ p: Point) -> (Double, Point, Point, Point, Point) {
    let (northState, northReward) = move(p, Action.north)
    let (southState, southReward) = move(p, Action.south)
    let (eastState, eastReward) = move(p, Action.east)
    let (westState, westReward) = move(p, Action.west)

    let const = prob * Double(northReward + southReward + eastReward + westReward)

    return (const, northState, southState, eastState, westState)
}

func make_figure_3_2() {
    let A = np.zeros([nStates, nStates])
    let b = np.zeros([nStates, 1])

    //Compute the Bellman equation for each state
    for i in 0..<nStates {
        let p = intToGrid(i)
        let (const, northState, southState, eastState, westState) = generateEquation(p)

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

    let vGrid = np.flipud(np.reshape(v, Python.tuple([n, n])))

    let fig = plt.figure(figsize: [6.4, 4.8])
    let ax = fig.gca()

    ax.imshow(vGrid, cmap: "YlGn", interpolation: "none")
    for i in 0..<n {
        for j in 0..<n {
            ax.text(j, i, np.around(vGrid[i, j], 1), ha: "center", va: "center", color: "b")
        }
    }
    ax.set_title("Gridworld state-value function")
    ax.tick_params(axis: "x",labelbottom: "off")
    ax.tick_params(axis: "y",labelleft: "off")
    plt.savefig("Fig_3.2.png")
}

make_figure_3_2()
