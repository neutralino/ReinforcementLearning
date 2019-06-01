//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-05-18.
//  Copyright © 2019 Don Teo. All rights reserved.
//

// import Foundation
import Python
PythonLibrary.useVersion(2)  // stuck with the System python for now. (i.e. /usr/bin/python -m pip)
//import TensorFlow

let plt = Python.import("matplotlib.pyplot")

let nRuns = 2000
let nTimeStepsPerRun = 1000

// greedy strategy
func getMaxActionValueEstimateIndex(avEstimate: [Int: Float]) -> Int {
    let maxVal = avEstimate.values.max()
    let optimalActions = avEstimate.filter { $0.value == maxVal }
    let optimalActionIndices = Array(optimalActions.keys)
    return optimalActionIndices.randomElement()!
}

// random strategy
func getRandomActionIndex(avEstimate: [Int: Float]) -> Int {
    return Array(avEstimate.keys).randomElement()!
}

// epsilon-greedy strategy
func getEpsilonGreedyIndex(avEstimate: [Int: Float], epsilon: Double) -> Int {
    let d = Double.random(in: 0...1)
    if d <= epsilon {
        return getRandomActionIndex(avEstimate: avEstimate)
    }
    return getMaxActionValueEstimateIndex(avEstimate: avEstimate)
}

func getEpsilonGreedyIndex1(avEstimate: [Int: Float]) -> Int {
    return getEpsilonGreedyIndex(avEstimate: avEstimate, epsilon: 0.1)
}

func getEpsilonGreedyIndex2(avEstimate: [Int: Float]) -> Int {
    return getEpsilonGreedyIndex(avEstimate: avEstimate, epsilon: 0.01)
}

class ActionTracker {
    // Row i stores the reward of each run for time step i
    var allRewards = Array(repeating: Array(repeating: 0.0, count: nRuns), count: nTimeStepsPerRun)

    var averageRewards = Array(repeating: 0.0, count: nTimeStepsPerRun)

    var actionTaken = Array(repeating: Array(repeating: -1, count: nRuns), count: nTimeStepsPerRun)
    var optimalAction = Array(repeating: [Int](), count: nRuns)
    var averageOptimal = Array(repeating: 0.0, count: nTimeStepsPerRun)

    // We need to keep track of the estimated value of each action.
    // Here, we use the sample average estimate.
    // These are reset at every run.
    var actionValueEstimate: [Int: Float] = [:]
    var actionCounter: [Int: Int] = [:]

    func computeAverageReward() {
        for iTimestep in 0...self.allRewards.count-1 {
            self.averageRewards[iTimestep] = self.allRewards[iTimestep].reduce(0, +) / Double(self.allRewards[iTimestep].count)
        }
    }

    func computeAverageOptimal() {
        for iTimeStep in 0...self.allRewards.count-1 {
            var nOptimalActionTaken = 0
            for iRun in 0...nRuns-1 {
                let optimal = self.optimalAction[iRun]
                if optimal.contains(self.actionTaken[iTimeStep][iRun]) {
                    nOptimalActionTaken += 1
                }
            }
            self.averageOptimal[iTimeStep] = Double(nOptimalActionTaken) / Double(nRuns)
        }
    }
}

func completeRun(priorActionTracker: ActionTracker,
                 iRun: Int,
                 strategy: ([Int: Float]) -> Int,
                 testbed: TenArmedTestbed) -> ActionTracker {
    let actionTracker = priorActionTracker
    actionTracker.optimalAction[iRun] = testbed.indicesOfOptimalAction

    // initialize
    for i in 0...testbed.arms.count-1 {
        actionTracker.actionValueEstimate[i] = 0
        actionTracker.actionCounter[i] = 0
    }

    for iTimeStep in 0...nTimeStepsPerRun-1 {
        // determine next action
        let actionIndex = strategy(actionTracker.actionValueEstimate)
        actionTracker.actionTaken[iTimeStep][iRun] = actionIndex
        actionTracker.actionCounter[actionIndex]! += 1

        let reward = testbed.arms[actionIndex].nextFloat()
        actionTracker.allRewards[iTimeStep][iRun] = Double(reward)

        let currentActionValue = actionTracker.actionValueEstimate[actionIndex]!
        let nextActionValue = currentActionValue + 1 / Float(actionTracker.actionCounter[actionIndex]!) * (reward - currentActionValue)
        // update action value estimate
        actionTracker.actionValueEstimate[actionIndex] = nextActionValue
    }

    return actionTracker
}

var greedyActionTracker = ActionTracker()
var epsilonGreedy1ActionTracker = ActionTracker()
var epsilonGreedy2ActionTracker = ActionTracker()

for iRun in 0...nRuns-1 {
    let testbed = TenArmedTestbed()
    //testbed.printStats()

    greedyActionTracker = completeRun(
        priorActionTracker: greedyActionTracker,
        iRun: iRun,
        strategy: getMaxActionValueEstimateIndex,
        testbed: testbed)

    epsilonGreedy1ActionTracker = completeRun(
        priorActionTracker: epsilonGreedy1ActionTracker,
        iRun: iRun,
        strategy: getEpsilonGreedyIndex1,
        testbed: testbed)

    epsilonGreedy2ActionTracker = completeRun(
        priorActionTracker: epsilonGreedy2ActionTracker,
        iRun: iRun,
        strategy: getEpsilonGreedyIndex2,
        testbed: testbed)

    //for (key,value) in actionValueEstimate {
    //    print("\(key) : \(value)")
    //}
}

// print(allRewards)
greedyActionTracker.computeAverageReward()
epsilonGreedy1ActionTracker.computeAverageReward()
epsilonGreedy2ActionTracker.computeAverageReward()

greedyActionTracker.computeAverageOptimal()
epsilonGreedy1ActionTracker.computeAverageOptimal()
epsilonGreedy2ActionTracker.computeAverageOptimal()

print("Greedy Action Average Rewards")
print(greedyActionTracker.averageRewards.suffix(10))
print("epsilon=0.1 Greedy Action Average Rewards")
print(epsilonGreedy1ActionTracker.averageRewards.suffix(10))
print("epsilon=0.01 Greedy Action Average Rewards")
print(epsilonGreedy2ActionTracker.averageRewards.suffix(10))

let fig = plt.figure(figsize: [10, 10])

plt.subplot(2, 1, 1)
plt.plot(Array(1...nTimeStepsPerRun), greedyActionTracker.averageRewards, label: "eps=0")
plt.plot(Array(1...nTimeStepsPerRun), epsilonGreedy1ActionTracker.averageRewards, label: "eps=0.1")
plt.plot(Array(1...nTimeStepsPerRun), epsilonGreedy2ActionTracker.averageRewards, label: "eps=0.01")
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.legend()

print("Greedy Action Optimal Fraction")
print(greedyActionTracker.averageOptimal.suffix(10))
print("epsilon=0.1 Greedy Action Optimal Fraction")
print(epsilonGreedy1ActionTracker.averageOptimal.suffix(10))
print("epsilon=0.01 Greedy Action Optimal Fraction")
print(epsilonGreedy2ActionTracker.averageOptimal.suffix(10))

plt.subplot(2, 1, 2)
plt.plot(Array(1...nTimeStepsPerRun), greedyActionTracker.averageOptimal, label: "eps=0")
plt.plot(Array(1...nTimeStepsPerRun), epsilonGreedy1ActionTracker.averageOptimal, label: "eps=0.1")
plt.plot(Array(1...nTimeStepsPerRun), epsilonGreedy2ActionTracker.averageOptimal, label: "eps=0.01")
plt.xlabel("Steps")
plt.ylabel("% Optimal award")
plt.legend()


plt.savefig("Fig_2.2.png")
