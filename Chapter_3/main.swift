//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-06-22.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import Python

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let linalg = Python.import("numpy.linalg")

let n = 5
let nStates = n * n
let gamma = 0.9
let prob = 0.25

func generateBellmanEquation(_ p: Point, _ gridWorld: GridWorld) -> (Double, Point, Point, Point, Point) {
    let (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward) = gridWorld.getNextStatesAndRewards(p)
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
    let gridWorld = GridWorld(n)

    let A = np.zeros([nStates, nStates])
    let b = np.zeros([nStates, 1])

    // Compute the Bellman equation for each state
    for i in 0..<nStates {
        let p = gridWorld.intToGrid(i)
        let (const, northState, southState, eastState, westState) = generateBellmanEquation(p, gridWorld)

        let coef = PythonObject(prob * gamma)

        // fill row in matrix
        A[i, gridWorld.gridToInt(northState)] += coef
        A[i, gridWorld.gridToInt(southState)] += coef
        A[i, gridWorld.gridToInt(eastState)] += coef
        A[i, gridWorld.gridToInt(westState)] += coef

        // move p to RHS
        A[i, gridWorld.gridToInt(p)] -= 1

        // move const to LHS
        b[i] = PythonObject(-const)
    }

    // solve for state-value function
    let v = linalg.solve(A, b)

    make_value_function_heatmap(valueFunction: v, title: "Gridworld state-value function", filename: "Fig_3.2.png")
}

func make_figure_3_5() {
    let gridWorld = GridWorld(n)

    //solve via value iteration (see Section 4.4)
    let valueFunction = np.zeros([nStates, 1])

    var delta = 0.0
    let minDelta = 1e-5
    var count = 0
    repeat {
        delta = 0.0
        count += 1
        for i in 0..<nStates {
            let p = gridWorld.intToGrid(i)

            let (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward) = gridWorld.getNextStatesAndRewards(p)

            // determine the best value by looping through all actions
            let expectedNorthActionValue = Double(northReward) + gamma * Double(valueFunction[gridWorld.gridToInt(northState)])!
            let expectedSouthActionValue = Double(southReward) + gamma * Double(valueFunction[gridWorld.gridToInt(southState)])!
            let expectedWestActionValue  = Double(westReward)  + gamma * Double(valueFunction[gridWorld.gridToInt(westState)])!
            let expectedEastActionValue  = Double(eastReward)  + gamma * Double(valueFunction[gridWorld.gridToInt(eastState)])!
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
