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
    let gridWorld = GridWorld(n)

    // perform iterative policy evaluation for the random policy
    let v = np.zeros([nStates, 1])
    var delta = 0.0
    let minDelta = 1e-5
    var count = 0
    repeat {
        delta = 0.0
        count += 1
        for i in 0..<nStates {
            let p = gridWorld.intToGrid(i)
            let (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward) = gridWorld.getNextStatesAndRewards(p)

            // determine the next iteration of the state's value
            let expectedNorthActionValue = Double(northReward) + gamma * Double(v[gridWorld.gridToInt(northState)])!
            let expectedSouthActionValue = Double(southReward) + gamma * Double(v[gridWorld.gridToInt(southState)])!
            let expectedWestActionValue  = Double(westReward)  + gamma * Double(v[gridWorld.gridToInt(westState)])!
            let expectedEastActionValue  = Double(eastReward)  + gamma * Double(v[gridWorld.gridToInt(eastState)])!
            let nextV = prob * (expectedNorthActionValue + expectedSouthActionValue + expectedWestActionValue + expectedEastActionValue)
            delta = max(delta, abs(Double(v[i])! - nextV))
            v[i] = PythonObject(nextV)
        }
        print("Finished \(count) iterations, delta = \(delta)")
    } while delta > minDelta
    make_value_function_heatmap(valueFunction: v, title: "Small gridworld state-value function", filename: "Fig_4.1.png")
}

make_figure_4_1()
