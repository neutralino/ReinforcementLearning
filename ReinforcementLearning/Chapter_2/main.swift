//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-05-18.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import Python
PythonLibrary.useVersion(2)  // stuck with the System python for now. (i.e. /usr/bin/python -m pip)

let plt = Python.import("matplotlib.pyplot")

let nRuns = 2000
let nTimeStepsPerRun = 1000

// greedy strategy
func getMaxActionValueEstimateIndex(avEstimate: [Int: Double]) -> Int {
    let maxVal = avEstimate.values.max()
    let optimalActions = avEstimate.filter { $0.value == maxVal }
    let optimalActionIndices = Array(optimalActions.keys)
    return optimalActionIndices.randomElement()!
}

// random strategy
func getRandomActionIndex(avEstimate: [Int: Double]) -> Int {
    return Array(avEstimate.keys).randomElement()!
}

// epsilon-greedy strategy
func getEpsilonGreedyIndex(avEstimate: [Int: Double], epsilon: Double) -> Int {
    let d = Double.random(in: 0...1)
    if d <= epsilon {
        return getRandomActionIndex(avEstimate: avEstimate)
    }
    return getMaxActionValueEstimateIndex(avEstimate: avEstimate)
}

// upper-confidence-bound strategy
func getUCBActionIndex(avEstimate: [Int: Double], actionCounter: [Int: Int], iTimeStep: Int, c: Double) -> Int {
    //if there are any actions that have yet to be taken, these are maximizing actions
    let optimalActions = actionCounter.filter { $0.value == 0 }
    if optimalActions.count > 0 {
        let optimalActionIndices = Array(optimalActions.keys)
        return optimalActionIndices.randomElement()!
    }

    var ucbAvEstimate = [Int: Double]()
    for index in avEstimate.keys {
        ucbAvEstimate[index] = avEstimate[index]! + c * sqrt(log(Double(iTimeStep)) / Double(actionCounter[index]!))
    }
    return getMaxActionValueEstimateIndex(avEstimate: ucbAvEstimate)
}

func getStrategyIndex(avEstimate: [Int: Double], epsilon: Double, c: Double,
                           actionCounter: [Int: Int], iTimeStep: Int) -> Int {
    if c > 0 {
        return getUCBActionIndex(avEstimate: avEstimate, actionCounter: actionCounter, iTimeStep: iTimeStep, c: c)
    }
    return getEpsilonGreedyIndex(avEstimate: avEstimate, epsilon: epsilon)
}

// soft-max distribution sampling strategy
func getGradientBanditActionIndex(energy: [Double]) -> (Int, [Double]) {
    let partitionFunction = energy.map { exp($0) }.reduce(0, +)
    let probDist = energy.map { exp($0) / partitionFunction }
    let discreteDistribution = DiscreteDistribution(randomSource: random, distribution: probDist)
    return (discreteDistribution.nextInt(), probDist)
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
    var actionValueEstimate: [Int: Double] = [:]
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

func simulate(actionTracker: ActionTracker,
                 iRun: Int,
                 epsilon: Double,
                 c: Double,
                 testbed: TenArmedTestbed,
                 initialActionValueEstimate: Double,
                 stepSize: Double) {
    actionTracker.optimalAction[iRun] = testbed.indicesOfOptimalAction

    // initialize
    for i in 0...testbed.arms.count-1 {
        actionTracker.actionValueEstimate[i] = initialActionValueEstimate
        actionTracker.actionCounter[i] = 0
    }

    for iTimeStep in 0...nTimeStepsPerRun-1 {
        // determine next action
        //let actionIndex = strategy(actionTracker.actionValueEstimate)
        let actionIndex = getStrategyIndex(avEstimate: actionTracker.actionValueEstimate,
                                           epsilon: epsilon,
                                           c: c,
                                           actionCounter: actionTracker.actionCounter,
                                           iTimeStep: iTimeStep)
        actionTracker.actionTaken[iTimeStep][iRun] = actionIndex
        actionTracker.actionCounter[actionIndex]! += 1

        let reward = Double(testbed.arms[actionIndex].nextFloat())
        actionTracker.allRewards[iTimeStep][iRun] = Double(reward)

        let currentActionValue = actionTracker.actionValueEstimate[actionIndex]!

        let theStepSize = stepSize > 0 ? stepSize : 1 / Double(actionTracker.actionCounter[actionIndex]!)
        let nextActionValue = currentActionValue + theStepSize * (reward - currentActionValue)
        // update action value estimate
        actionTracker.actionValueEstimate[actionIndex] = nextActionValue
    }
}

func simulateAll(actionTrackers: [ActionTracker], epsilons: [Double],
                 cs: [Double],
                 initialActionValueEstimates: [Double],
                 stepSize: Double = -1.0) {
    for iRun in 0...nRuns-1 {
        let testbed = TenArmedTestbed(mean: 0)

        for (((actionTracker, c), epsilon), initialActionValueEstimate) in zip(zip(zip(actionTrackers, cs), epsilons), initialActionValueEstimates) {
            simulate(actionTracker: actionTracker, iRun: iRun, epsilon: epsilon, c: c, testbed: testbed,
                     initialActionValueEstimate: initialActionValueEstimate, stepSize: stepSize)
        }
    }
    for actionTracker in actionTrackers {
        actionTracker.computeAverageReward()
        actionTracker.computeAverageOptimal()
    }
}

func make_figure_2_2() {
    let greedyActionTracker = ActionTracker()
    let epsilonGreedy1ActionTracker = ActionTracker()
    let epsilonGreedy2ActionTracker = ActionTracker()

    let actionTrackers = [greedyActionTracker, epsilonGreedy1ActionTracker, epsilonGreedy2ActionTracker]
    let epsilons = [0, 0.1, 0.01]
    let initialActionValueEstimates = [0.0, 0.0, 0.0]
    let cs = [-1.0, -1.0, -1.0]
    simulateAll(actionTrackers: actionTrackers, epsilons: epsilons, cs: cs, initialActionValueEstimates: initialActionValueEstimates)

    let _ = plt.figure(figsize: [10, 10])

    plt.subplot(2, 1, 1)
    let labels = ["eps=0", "eps=0.1", "eps=0.01"]
    for (actionTracker, label) in zip(actionTrackers, labels) {
        plt.plot(Array(1...nTimeStepsPerRun), actionTracker.averageRewards, label: label)
    }
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    for (actionTracker, label) in zip(actionTrackers, labels) {
        plt.plot(Array(1...nTimeStepsPerRun), actionTracker.averageOptimal, label: label)
    }
    plt.xlabel("Steps")
    plt.ylabel("% Optimal award")
    plt.legend()

    plt.savefig("Fig_2.2.png")
}

func make_figure_2_3() {
    let epsilonGreedy1ActionConstantStepTracker = ActionTracker()
    let optimisticGreedyActionConstantStepTracker = ActionTracker()

    let actionTrackers = [epsilonGreedy1ActionConstantStepTracker, optimisticGreedyActionConstantStepTracker]
    let epsilons = [0.1, 0]
    let cs = [-1.0, -1.0]
    let stepSize = 0.1
    let initialActionValueEstimates = [0.0, 5.0]
    simulateAll(actionTrackers: actionTrackers, epsilons: epsilons, cs: cs,
                initialActionValueEstimates: initialActionValueEstimates,
                stepSize: stepSize)

    let _ = plt.figure(figsize: [6.4, 4.8])

    plt.plot(Array(1...nTimeStepsPerRun), actionTrackers[0].averageOptimal, label: "Q1=0, eps=0.1")
    plt.plot(Array(1...nTimeStepsPerRun), actionTrackers[1].averageOptimal, label: "Q1=5, eps=0")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal award")
    plt.legend()

    plt.savefig("Fig_2.3.png")
}

func make_figure_2_4() {
    let epsilonGreedy1ActionTracker = ActionTracker()
    let ucbActionTracker = ActionTracker()

    let actionTrackers = [epsilonGreedy1ActionTracker, ucbActionTracker]
    let epsilons = [0.1, 0.0]
    let initialActionValueEstimates = [0.0, 0.0]
    let cs = [-1.0, 2.0]
    simulateAll(actionTrackers: actionTrackers, epsilons: epsilons, cs: cs, initialActionValueEstimates: initialActionValueEstimates)

    let _ = plt.figure(figsize: [6.4, 4.8])

    let labels = ["eps-greedy eps=0.1", "UCB c=2"]
    for (actionTracker, label) in zip(actionTrackers, labels) {
        plt.plot(Array(1...nTimeStepsPerRun), actionTracker.averageRewards, label: label)
    }
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()

    plt.savefig("Fig_2.4.png")
}


func gradientBanditSimulate(actionTracker: ActionTracker,
                            iRun: Int,
                            alpha: Double,
                            testbed: TenArmedTestbed,
                            withBaseline: Bool) {
    actionTracker.optimalAction[iRun] = testbed.indicesOfOptimalAction
    var energy = Array(repeating: 0.0, count: testbed.arms.count)
    var averageReward = 0.0
    for iTimeStep in 0...nTimeStepsPerRun-1 {
        let (actionIndex, probDist) = getGradientBanditActionIndex(energy: energy)

        actionTracker.actionTaken[iTimeStep][iRun] = actionIndex
        let reward = Double(testbed.arms[actionIndex].nextFloat())
        averageReward = averageReward + (1 / Double(iTimeStep + 1)) * (reward - averageReward)
        actionTracker.allRewards[iTimeStep][iRun] = Double(reward)

        let baseline = withBaseline ? averageReward : 0

        for i in 0...testbed.arms.count-1 {
            if i == actionIndex {
                energy[i] += alpha * (reward - baseline) * (1 - probDist[i])
            }
            else {
                energy[i] -= alpha * (reward - baseline) * probDist[i]
            }
        }
    }
}

func gradientBanditSimulateAll(actionTrackers: [ActionTracker],
                               alphas: [Double],
                               withBaselineFlags: [Bool],
                               mean: Float) {
    for iRun in 0...nRuns-1 {
        let testbed = TenArmedTestbed(mean: mean)

        for ((actionTracker, alpha), withBaseline) in zip(zip(actionTrackers, alphas), withBaselineFlags) {
            gradientBanditSimulate(actionTracker: actionTracker, iRun: iRun, alpha: alpha, testbed: testbed,
                                   withBaseline: withBaseline)

        }
    }
    for actionTracker in actionTrackers {
        actionTracker.computeAverageReward()
        actionTracker.computeAverageOptimal()
    }
}

func make_figure_2_5() {
    let gradientBasedActionTracker1 = ActionTracker()
    let gradientBasedActionTracker2 = ActionTracker()
    let gradientBasedActionTracker3 = ActionTracker()
    let gradientBasedActionTracker4 = ActionTracker()
    let actionTrackers = [gradientBasedActionTracker1, gradientBasedActionTracker2,
                          gradientBasedActionTracker3, gradientBasedActionTracker4]
    let alphas = [0.1, 0.4, 0.1, 0.4]
    let withBaselineFlags = [false, false, true, true]
    gradientBanditSimulateAll(actionTrackers: actionTrackers, alphas: alphas,
                              withBaselineFlags: withBaselineFlags, mean: 4)

    let _ = plt.figure(figsize: [6.4, 4.8])

    plt.plot(Array(1...nTimeStepsPerRun), actionTrackers[0].averageOptimal, label: "alpha=0.1 (w/o baseline)")
    plt.plot(Array(1...nTimeStepsPerRun), actionTrackers[1].averageOptimal, label: "alpha=0.4 (w/o baseline)")
    plt.plot(Array(1...nTimeStepsPerRun), actionTrackers[2].averageOptimal, label: "alpha=0.1 (w/ baseline)")
    plt.plot(Array(1...nTimeStepsPerRun), actionTrackers[3].averageOptimal, label: "alpha=0.4 (w/ baseline)")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal award")
    plt.legend()

    plt.savefig("Fig_2.5.png")
}

make_figure_2_2()
make_figure_2_3()
make_figure_2_4()
make_figure_2_5()
