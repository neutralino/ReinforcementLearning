//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-06-22.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import LANumerics
import SwiftPlot
import AGGRenderer

let n = 5
let nStates = n * n
let gamma = 0.9
let prob = 0.25

func generateBellmanEquation(_ p: Point, _ gridWorld: GridWorld) -> (Double, Point, Point, Point, Point) {
    let (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward) = gridWorld.getNextStatesAndRewards(p)
    let const = prob * Double(northReward + southReward + eastReward + westReward)
    return (const, northState, southState, eastState, westState)
}

func make_value_function_heatmap(valueFunction: Vector<Double>, title: String, filename: String) {
    var valueFunctionArray = [[Double]]()
    for i in (0..<n).reversed() {
        var row = [Double]()
        for j in 0..<n {
            let k = i * n + j
            row.append(valueFunction[k])
        }
        valueFunctionArray.append(row)
    }

    let aggRenderer: AGGRenderer = AGGRenderer()
    var heatmap = Heatmap<[[Double]]>(valueFunctionArray)
    heatmap.plotTitle.title = title
    try? heatmap.drawGraphAndOutput(fileName: filename, renderer: aggRenderer)
}

func make_figure_3_2() {
    let gridWorld = GridWorld(n)

    var A = Matrix<Double>(rows: Array(repeating: Array(repeating: 0.0, count: nStates), count: nStates))
    var b: Vector<Double> = Array(repeating: 0.0, count: nStates)

    // Compute the Bellman equation for each state
    for i in 0..<nStates {
        let p = gridWorld.intToGrid(i)
        let (const, northState, southState, eastState, westState) = generateBellmanEquation(p, gridWorld)

        let coef = prob * gamma

        // fill row in matrix
        A[i, gridWorld.gridToInt(northState)] += coef
        A[i, gridWorld.gridToInt(southState)] += coef
        A[i, gridWorld.gridToInt(eastState)] += coef
        A[i, gridWorld.gridToInt(westState)] += coef

        // move p to RHS
        A[i, gridWorld.gridToInt(p)] -= 1

        // move const to LHS
        b[i] = -const
    }

    // solve for state-value function
    let v = A.solve(b)!
    make_value_function_heatmap(valueFunction: v, title: "Gridworld state-value function", filename: "Output/Chapter3/Fig_3.2")
}

func make_figure_3_5() {
    let gridWorld = GridWorld(n)

    // solve via value iteration (see Section 4.4)
    var valueFunction: Vector<Double> = Array(repeating: 0.0, count: nStates)

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
            let expectedNorthActionValue: Double = Double(northReward) + gamma * valueFunction[gridWorld.gridToInt(northState)]
            let expectedSouthActionValue: Double = Double(southReward) + gamma * valueFunction[gridWorld.gridToInt(southState)]
            let expectedWestActionValue: Double = Double(westReward) + gamma * valueFunction[gridWorld.gridToInt(westState)]
            let expectedEastActionValue: Double = Double(eastReward) + gamma * valueFunction[gridWorld.gridToInt(eastState)]
            let nextV = max(expectedNorthActionValue, expectedSouthActionValue, expectedWestActionValue, expectedEastActionValue)

            delta = max(delta, abs(valueFunction[i] - nextV))
            valueFunction[i] = nextV
        }
        print("Finished \(count) iterations, delta = \(delta)")
    } while delta > minDelta

    print(valueFunction)
    make_value_function_heatmap(valueFunction: valueFunction, title: "Gridworld optimal state-value function", filename: "Output/Chapter3/Fig_3.5")
}

make_figure_3_2()
make_figure_3_5()
