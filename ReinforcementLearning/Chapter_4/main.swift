//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-08-04.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import Python
import GameplayKit

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")
let linalg = Python.import("numpy.linalg")

func make_value_function_heatmap(n: Int, valueFunction: PythonObject, title: String, filename: String, cellText: Bool = true) {
    let vGrid = np.reshape(valueFunction, Python.tuple([n, n]))
    let fig = plt.figure(figsize: [6.4, 4.8])
    let ax = fig.gca()

    if cellText {
        ax.imshow(vGrid, cmap: "YlGn", interpolation: "none")
        for i in 0..<n {
            for j in 0..<n {
                ax.text(j, i, np.around(vGrid[i, j], 1), ha: "center", va: "center", color: "b")
            }
        }
    }

    ax.set_title(title)
    ax.tick_params(axis: "x",labelbottom: "off")
    ax.tick_params(axis: "y",labelleft: "off")
    plt.savefig(filename)
}

func make_figure_4_1() {
    let n = 4
    let nStates = n * n
    let gamma = 1.0
    let prob = 0.25

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
    make_value_function_heatmap(n: n, valueFunction: v, title: "Small gridworld state-value function", filename: "Fig_4.1.png")
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
                               policy: [[Int]],
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
              action: action,
              v: v,
              carRental: carRental
            )
            delta = max(delta, abs(v[state1][state2] - nextV))
            v[state1][state2] = nextV
        }
        print("Finished \(count) iterations, delta = \(delta)")
    } while delta > minDelta
}

func run_policy_improvement_4_2(policy: inout [[Int]], v: [[Double]],
                                n: Int,
                                nStates: Int,
                                carRental: CarRental) -> Bool {
    print("Running policy improvement")
    var isStable = true
    for i in 0..<nStates {
        let state1 = i % n
        let state2 = i / n

        let oldAction = policy[state1][state2]

        let actionSpace = Array(-5...5)
        let expectedReturns = actionSpace.map{ computeExpectedReturn(
                                             state1: state1,
                                             state2: state2,
                                             action: $0,
                                             v: v,
                                             carRental: carRental
                                           ) }
        let maxExpectedReturnIndex = expectedReturns.firstIndex{ $0 == expectedReturns.max() }!
        let bestAction = actionSpace[maxExpectedReturnIndex]

        if bestAction != oldAction {
            isStable = false
        }
        policy[state1][state2] = bestAction
    }
    return isStable
}

// func convertIntArrayToNumpy(array: [[Int]]) -> PythonObject {
//     let n = array.count
//     let numpyArray = np.zeros([n, n])
//     for i in 0..<n {
//         for j in 0..<n {
//             numpyArray[i][j] = PythonObject(array[i][j])
//         }
//     }
//     return numpyArray
// }

func make_figure_4_2() {
    let n = 21
    let nStates = n * n
    let initialState = (10, 10)
    // the initial policy is one that never moves any cars
    var policy = Array(repeating: Array(repeating: 0, count: n), count: n)

    // the initial value function is 0 for all states
    var v = Array(repeating: Array(repeating: 0.0, count: n), count: n)

    let carRental = CarRental(
      maxCars: n-1,
      request1Mean: 3,
      request2Mean: 4,
      return1Mean: 3,
      return2Mean: 2,
      initialState: initialState)

    let figAndAx = plt.subplots(nrows: 2, ncols: 3).tuple2
    let fig = figAndAx.0
    let ax = figAndAx.1

    var count = 0
    var im = ax[0][0].imshow(policy, cmap: "YlGn", interpolation: "none", origin: "lower")
    ax[0][0].get_xaxis().set_visible(false)
    ax[0][0].get_yaxis().set_visible(false)
    ax[0][0].set_title("pi_\(count)")
    var isStable = false
    repeat {
        print("Policy iteration #\(count+1)")
        count += 1
        // run policy evaluation
        run_policy_evaluation_4_2(v: &v, policy: policy, n: n, nStates: nStates, carRental: carRental)
        // run policy improvement
        isStable = run_policy_improvement_4_2(policy: &policy, v: v, n: n, nStates: nStates, carRental: carRental)

        if count < 6 {
            let index0 = count / 3
            let index1 = count % 3
            im = ax[index0][index1].imshow(policy, cmap: "YlGn",
                                      interpolation: "none",
                                      origin: "lower"
            )
            if index0 == 1 && index1 == 0 {
                ax[index0][index1].set_xlabel("# Cars 2nd location")
                ax[index0][index1].set_ylabel("# Cars 1st location")

            }
            else {
                ax[index0][index1].get_xaxis().set_visible(false)
                ax[index0][index1].get_yaxis().set_visible(false)
            }
            ax[index0][index1].set_title("pi_\(count)")
        }
    } while !isStable

    let imV = ax[1][2].imshow(v, cmap: "Blues", interpolation: "none", origin: "lower")
    ax[1][2].get_xaxis().set_visible(false)
    ax[1][2].get_yaxis().set_visible(false)
    ax[1][2].set_title("v_pi_\(count)")

    //show the policy colorbar first
    fig.colorbar(imV, ax: ax.ravel().tolist())
    fig.colorbar(im, ax: ax.ravel().tolist())

    plt.savefig("Fig_4.2.png")
}

func run_value_iteration_4_3(v: inout [Double],
                             gamblersProblem: GamblersProblem) -> [[Double]]{
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
        print(bestActions)
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
    print(v.count)
    let vIterations = run_value_iteration_4_3(v: &v, gamblersProblem: gamblersProblem)

    // get corresponding policy
    let policy = get_policy_from_value(v: vIterations.last!, gamblersProblem: gamblersProblem)
    print(policy)

    let figAndAx = plt.subplots(nrows: 2, ncols: 1, figsize: [6.4, 6.4]).tuple2
    let ax = figAndAx.1
    let xPoints = Array(1...n)
    ax[0].plot(xPoints, vIterations[1], label: "sweep 1")
    ax[0].plot(xPoints, vIterations[2], label: "sweep 2")
    ax[0].plot(xPoints, vIterations[3], label: "sweep 3")
    ax[0].plot(xPoints, vIterations.last, label: "final sweep")
    ax[0].legend()
    ax[0].set_xlabel("Capital")
    ax[0].set_ylabel("Value Estimate")

    ax[1].bar(xPoints, policy)
    ax[1].set_xlabel("Capital")
    ax[1].set_ylabel("Final Policy (stake)")

    plt.savefig("Fig_4.3.png")
}

make_figure_4_1()
make_figure_4_2()
make_figure_4_3()
