//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-08-04.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import GameplayKit
import LANumerics
import SwiftPlot
import AGGRenderer

func convert_vector_to_2D_array(n: Int, vector: Vector<Double>) -> [[Double]] {
    var twoDimArray = [[Double]]()
    for i in (0..<n).reversed() {
        var row = [Double]()
        for j in 0..<n {
            let k = i * n + j
            row.append(vector[k])
        }
        twoDimArray.append(row)
    }
    return twoDimArray
}

func make_heatmap_from_vector(n: Int, vector: Vector<Double>, title: String, filename: String?) -> Heatmap<[[Double]]> {
    let twoDimArray = convert_vector_to_2D_array(n: n, vector: vector)
    return make_heatmap(n: n, array: twoDimArray, title: title, filename: filename)
}

func make_heatmap(n: Int, array: [[Double]], title: String, filename: String?) -> Heatmap<[[Double]]> {
    let aggRenderer: AGGRenderer = AGGRenderer()
    var heatmap = Heatmap<[[Double]]>(array)
    heatmap.plotTitle.title = title
    if filename != nil {
        try? heatmap.drawGraphAndOutput(fileName: filename!, renderer: aggRenderer)
    }
    return heatmap
}

func make_figure_4_1() {
    let n = 4
    let nStates = n * n
    let gamma = 1.0
    let prob = 0.25

    let gridWorld = GridWorld(n)

    // perform iterative policy evaluation for the random policy
    var v: Vector<Double> = Array(repeating: 0.0, count: nStates)
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
            let expectedNorthActionValue: Double = Double(northReward) + gamma * v[gridWorld.gridToInt(northState)]
            let expectedSouthActionValue: Double = Double(southReward) + gamma * v[gridWorld.gridToInt(southState)]
            let expectedWestActionValue: Double = Double(westReward)  + gamma * v[gridWorld.gridToInt(westState)]
            let expectedEastActionValue: Double = Double(eastReward)  + gamma * v[gridWorld.gridToInt(eastState)]
            let nextV = prob * (expectedNorthActionValue + expectedSouthActionValue + expectedWestActionValue + expectedEastActionValue)
            delta = max(delta, abs(v[i] - nextV))
            v[i] = nextV
        }
        print("Finished \(count) iterations, delta = \(delta)")
    } while delta > minDelta
    make_heatmap_from_vector(n: n, vector: v, title: "Small gridworld state-value function", filename: "Output/Chapter4/Fig_4.1")
}

let MAX_POISSON_DIST_VALUE = 15

func computeExpectedReturn(state1: Int,
                           state2: Int,
                           action: Int,
                           v: [[Double]],
                           carRental: CarRental) -> Double {
    let gamma = 0.9
    var V = 0.0

    for jRequest1 in 0..<MAX_POISSON_DIST_VALUE {
        for kRequest2 in 0..<MAX_POISSON_DIST_VALUE {
            for lReturn1 in 0..<MAX_POISSON_DIST_VALUE {
                for mReturn2 in 0..<MAX_POISSON_DIST_VALUE {
                    let prob = carRental.RequestDistLoc1.prob(n: jRequest1) *
                      carRental.RequestDistLoc2.prob(n: kRequest2) *
                      carRental.ReturnDistLoc1.prob(n: lReturn1) *
                      carRental.ReturnDistLoc2.prob(n: mReturn2)
                    let nextStateAndReward: (state1: Int, state2: Int, reward: Int) = carRental.computeNextStateAndReward(
                      state1: state1,
                      state2: state2,
                      action: action,
                      request1: jRequest1,
                      request2: kRequest2,
                      return1: lReturn1,
                      return2: mReturn2)
                    V += prob * (Double(nextStateAndReward.reward) + gamma * v[nextStateAndReward.state1][nextStateAndReward.state2])
                }
            }
        }
    }
    return V
}

func run_policy_evaluation_4_2(v: inout [[Double]],
                               policy: [[Double]],
                               n: Int,
                               nStates: Int,
                               carRental: CarRental) {
    print("Running policy evaluation")
    var delta = 0.0
    let minDelta = 1e-5
    var count = 0
    repeat {
        delta = 0.0
        count += 1
        for i in 0..<nStates {
            let state1 = i % n
            let state2 = i / n
            let action = policy[state1][state2]
            // here we scan through all possible number of requests and returns
            // (up to some reasonable value) to compute the value function
            let nextV = computeExpectedReturn(
              state1: state1,
              state2: state2,
              action: Int(action),
              v: v,
              carRental: carRental
            )
            delta = max(delta, abs(v[state1][state2] - nextV))
            v[state1][state2] = nextV
        }
        print("Finished \(count) iterations, delta = \(delta)")
    } while delta > minDelta
}

func run_policy_improvement_4_2(policy: inout [[Double]], v: [[Double]],
                                n: Int,
                                nStates: Int,
                                carRental: CarRental) -> Bool {
    print("Running policy improvement")
    var isStable = true
    for i in 0..<nStates {
        let state1 = i % n
        let state2 = i / n

        let oldAction = policy[state1][state2]

        let actionSpace = Array(stride(from: -5, through: 5, by: 1.0))
        let expectedReturns = actionSpace.map { computeExpectedReturn(
                                             state1: state1,
                                             state2: state2,
                                             action: Int($0),
                                             v: v,
                                             carRental: carRental
                                           ) }
        let maxExpectedReturnIndex = expectedReturns.firstIndex { $0 == expectedReturns.max() }!
        let bestAction = actionSpace[maxExpectedReturnIndex]

        if bestAction != oldAction {
            isStable = false
        }
        policy[state1][state2] = bestAction
    }
    return isStable
}

func make_figure_4_2() {
    let n = 21
    let nStates = n * n
    let initialState = (10, 10)
    // the initial policy is one that never moves any cars
    var policy = Array(repeating: Array(repeating: 0.0, count: n), count: n)

    // the initial value function is 0 for all states
    var v = Array(repeating: Array(repeating: 0.0, count: n), count: n)

    let carRental = CarRental(
      maxCars: n-1,
      request1Mean: 3,
      request2Mean: 4,
      return1Mean: 3,
      return2Mean: 2,
      initialState: initialState)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 2, columns: 3))

    var count = 0

    var firstPlot = make_heatmap(n: n, array: policy, title: "pi_\(count)", filename: nil)
    firstPlot.plotLabel.xLabel = "# Cars 2nd location"
    firstPlot.plotLabel.yLabel = "# Cars 1st location"

    var plots = [firstPlot]

    var isStable = false
    repeat {
        print("Policy iteration #\(count+1)")
        count += 1
        // run policy evaluation
        run_policy_evaluation_4_2(v: &v, policy: policy, n: n, nStates: nStates, carRental: carRental)
        // run policy improvement
        isStable = run_policy_improvement_4_2(policy: &policy, v: v, n: n, nStates: nStates, carRental: carRental)

        if count < 5 {
            plots.append(make_heatmap(n: n, array: policy, title: "pi_\(count)", filename: nil))
        }
    } while !isStable

    plots.append(make_heatmap(n: n, array: v, title: "v_pi_\(count)", filename: nil))
    subplot.plots = plots
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter4/Fig_4.2", renderer: aggRenderer)
}

func run_value_iteration_4_3(v: inout [Double],
                             gamblersProblem: GamblersProblem) -> [[Double]] {
    var vIterations = [[Double]]()
    vIterations.append(Array(v))
    print("Running value iteration")
    var delta = 0.0
    let minDelta = 1e-6
    var count = 0
    repeat {
        delta = 0.0
        count += 1
        for i in 0..<gamblersProblem.n {
            let nextV = gamblersProblem.nextValue(state: i+1, v: v).0
            delta = max(delta, abs(v[i] - nextV))
            v[i] = nextV
        }
        print("Finished \(count) iterations, delta = \(delta)")
        vIterations.append(Array(v))
    } while delta > minDelta
    return vIterations
}

func get_policy_from_value(v: [Double],
                           gamblersProblem: GamblersProblem) -> [Int] {
    var policy = Array(repeating: -1, count: v.count)
    for i in 0..<policy.count {
        let bestActions = gamblersProblem.nextValue(state: i+1, v: v).1

        // we don't want to ever select the null action (pointless action)
        policy[i] = bestActions.filter { $0 != 0 }.first!
    }
    return policy
}

func make_figure_4_3() {
    let n = 99
    let gamblersProblem = GamblersProblem(n: n, pHead: 0.4)

    // initial value function for states 1 through 100
    // v[0] = state 1 ($1 capital)
    // ...
    // v[98] = state 99 ($99 capital)
    // (states 0 and 100 are terminal and have 0 value)
    var v = Array(repeating: 0.0, count: n)
    let vIterations = run_value_iteration_4_3(v: &v, gamblersProblem: gamblersProblem)

    // get corresponding policy
    let policy = get_policy_from_value(v: vIterations.last!, gamblersProblem: gamblersProblem)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 2, columns: 1))
    let xPoints = Array(stride(from: 1, through: Double(n), by: 1.0))
    var lineGraph = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)
    lineGraph.addSeries(xPoints, vIterations[1], label: "sweep 1", color: Color.blue)
    lineGraph.addSeries(xPoints, vIterations[2], label: "sweep 2", color: Color.orange)
    lineGraph.addSeries(xPoints, vIterations[3], label: "sweep 3", color: Color.green)
    lineGraph.addSeries(xPoints, vIterations.last!, label: "final sweep", color: Color.red)
    lineGraph.plotLabel.xLabel = "Capital"
    lineGraph.plotLabel.yLabel = "Value Estimate"

    var barGraph = BarGraph<Double, Double>()
    barGraph.addSeries(xPoints, policy.map { Double($0) }, label: "final policy")
    barGraph.plotLabel.xLabel = "Capital"
    barGraph.plotLabel.yLabel = "Final Policy (stake)"

    subplot.plots = [barGraph, lineGraph]
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter4/Fig_4.3", renderer: aggRenderer)
}

make_figure_4_1()
make_figure_4_2()
make_figure_4_3()
