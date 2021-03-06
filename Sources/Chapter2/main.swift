//
//  main.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-05-18.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import SwiftPlot
import AGGRenderer

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
    // if there are any actions that have yet to be taken, these are maximizing actions
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

    var meanAverageReward: Double = 0

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

    func computeMeanAverageReward() {
        self.meanAverageReward = Double(self.averageRewards.reduce(0, +)) / Double(self.averageRewards.count)
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
    for iAction in 0...testbed.arms.count-1 {
        actionTracker.actionValueEstimate[iAction] = initialActionValueEstimate
        actionTracker.actionCounter[iAction] = 0
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
    print("Generating Figure 2.2...")
    let greedyActionTracker = ActionTracker()
    let epsilonGreedy1ActionTracker = ActionTracker()
    let epsilonGreedy2ActionTracker = ActionTracker()

    let actionTrackers = [greedyActionTracker, epsilonGreedy1ActionTracker, epsilonGreedy2ActionTracker]
    let epsilons = [0, 0.1, 0.01]
    let initialActionValueEstimates = [0.0, 0.0, 0.0]
    let cs = [-1.0, -1.0, -1.0]
    simulateAll(actionTrackers: actionTrackers, epsilons: epsilons, cs: cs, initialActionValueEstimates: initialActionValueEstimates)

    let labels = ["eps=0", "eps=0.1", "eps=0.01"]
    let colors = [Color.blue, Color.orange, Color.green]
    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .vertical)

    var lineGraph1 = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)
    for (actionTracker, (color, label)) in zip(actionTrackers, zip(colors, labels)) {
        lineGraph1.addSeries(Array(1...nTimeStepsPerRun).map { Double($0) }, actionTracker.averageRewards, label: label, color: color)
    }
    lineGraph1.plotLabel.xLabel = "Steps"
    lineGraph1.plotLabel.yLabel = "Average reward"

    var lineGraph2 = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)
    for (actionTracker, (color, label)) in zip(actionTrackers, zip(colors, labels)) {
        lineGraph2.addSeries(Array(1...nTimeStepsPerRun).map { Double($0) }, actionTracker.averageOptimal, label: label, color: color)
    }
    lineGraph2.plotLabel.xLabel = "Steps"
    lineGraph2.plotLabel.yLabel = "% Optimal award"

    subplot.plots = [lineGraph1, lineGraph2]
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter2/Fig_2.2", renderer: aggRenderer)
}

func make_figure_2_3() {
    print("Generating Figure 2.3...")
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

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)
    lineGraph.addSeries(Array(1...nTimeStepsPerRun).map { Double($0) }, actionTrackers[0].averageOptimal, label: "Q1=0, eps=0.1", color: Color.blue)
    lineGraph.addSeries(Array(1...nTimeStepsPerRun).map { Double($0) }, actionTrackers[1].averageOptimal, label: "Q1=5, eps=0", color: Color.orange)
    lineGraph.plotLabel.xLabel = "Steps"
    lineGraph.plotLabel.yLabel = "% Optimal award"
    try? lineGraph.drawGraphAndOutput(fileName: "output/Chapter2/Fig_2.3", renderer: aggRenderer)
}

func make_figure_2_4() {
    print("Generating Figure 2.4...")
    let epsilonGreedy1ActionTracker = ActionTracker()
    let ucbActionTracker = ActionTracker()

    let actionTrackers = [epsilonGreedy1ActionTracker, ucbActionTracker]
    let epsilons = [0.1, 0.0]
    let initialActionValueEstimates = [0.0, 0.0]
    let cs = [-1.0, 2.0]
    simulateAll(actionTrackers: actionTrackers, epsilons: epsilons, cs: cs, initialActionValueEstimates: initialActionValueEstimates)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)

    let labels = ["eps-greedy eps=0.1", "UCB c=2"]
    let colors = [Color.blue, Color.orange]
    for (actionTracker, (label, color)) in zip(actionTrackers, zip(labels, colors)) {
        lineGraph.addSeries(Array(1...nTimeStepsPerRun).map { Double($0) }, actionTracker.averageRewards, label: label, color: color)
    }
    lineGraph.plotLabel.xLabel = "Steps"
    lineGraph.plotLabel.yLabel = "Average reward"
    try? lineGraph.drawGraphAndOutput(fileName: "output/Chapter2/Fig_2.4", renderer: aggRenderer)
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
        actionTracker.allRewards[iTimeStep][iRun] = Double(reward)

        averageReward += (1 / Double(iTimeStep + 1)) * (reward - averageReward)
        let baseline = withBaseline ? averageReward : 0

        for iAction in 0...testbed.arms.count-1 {
            if iAction == actionIndex {
                energy[iAction] += alpha * (reward - baseline) * (1 - probDist[iAction])
            } else {
                energy[iAction] -= alpha * (reward - baseline) * probDist[iAction]
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
    print("Generating Figure 2.5...")
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

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)

    lineGraph.addSeries(Array(1...nTimeStepsPerRun).map{ Double($0) }, actionTrackers[0].averageOptimal, label: "alpha=0.1 (w/o baseline)", color: Color.blue)
    lineGraph.addSeries(Array(1...nTimeStepsPerRun).map{ Double($0) }, actionTrackers[1].averageOptimal, label: "alpha=0.4 (w/o baseline)", color: Color.orange)
    lineGraph.addSeries(Array(1...nTimeStepsPerRun).map{ Double($0) }, actionTrackers[2].averageOptimal, label: "alpha=0.1 (w/ baseline)" , color: Color.green)
    lineGraph.addSeries(Array(1...nTimeStepsPerRun).map{ Double($0) }, actionTrackers[3].averageOptimal, label: "alpha=0.4 (w/ baseline)" , color: Color.red)
    lineGraph.plotLabel.xLabel = "Steps"
    lineGraph.plotLabel.yLabel = "% Optimal award"
    try? lineGraph.drawGraphAndOutput(fileName: "output/Chapter2/Fig_2.5", renderer: aggRenderer)
}

func make_figure_2_6() {
    print("Generating Figure 2.6...")

    // greedy with optimistic init
    var greedyOptimActionTrackers = [ActionTracker]()
    let optInitialActionValueEstimates = [1.0/4.0, 1.0/2.0, 1.0, 2.0, 4.0]
    let greedyEpsilons = Array(repeating: 0.0, count: optInitialActionValueEstimates.count)
    let greedyCs = Array(repeating: -1.0, count: greedyEpsilons.count)
    let stepSize = 0.1
    for _ in greedyEpsilons {
        greedyOptimActionTrackers.append(ActionTracker())
    }
    simulateAll(actionTrackers: greedyOptimActionTrackers, epsilons: greedyEpsilons, cs: greedyCs,
                initialActionValueEstimates: optInitialActionValueEstimates, stepSize: stepSize)

    var greedyOptimMeanAverageReward = [Double]()
    for actionTracker in greedyOptimActionTrackers {
        actionTracker.computeMeanAverageReward()
        greedyOptimMeanAverageReward.append(actionTracker.meanAverageReward)
    }

    // epsilon greedy
    var epsilonGreedyActionTrackers = [ActionTracker]()
    let epsilons = [1.0/128.0, 1.0/64.0, 1.0/32.0,  1.0/16.0, 1.0/8.0, 1.0/4.0]
    let cs = Array(repeating: -1.0, count: epsilons.count)
    let initialActionValueEstimates = Array(repeating: 0.0, count: epsilons.count)
    for _ in epsilons {
        epsilonGreedyActionTrackers.append(ActionTracker())
    }
    simulateAll(actionTrackers: epsilonGreedyActionTrackers, epsilons: epsilons, cs: cs, initialActionValueEstimates: initialActionValueEstimates)

    var epsilonGreedyMeanAverageReward = [Double]()
    for actionTracker in epsilonGreedyActionTrackers {
        actionTracker.computeMeanAverageReward()
        epsilonGreedyMeanAverageReward.append(actionTracker.meanAverageReward)
    }

    // UCB
    var ucbActionTrackers = [ActionTracker]()
    let ucbCs = [1.0/16.0, 1.0/8.0, 1.0/4.0, 1.0/2.0, 1.0, 2.0, 4.0]
    let ucbEpsilons = Array(repeating: 0.0, count: ucbCs.count)
    let ucbInitialActionValueEstimates = Array(repeating: 0.0, count: ucbCs.count)
    for _ in ucbCs {
        ucbActionTrackers.append(ActionTracker())
    }
    simulateAll(actionTrackers: ucbActionTrackers, epsilons: ucbEpsilons, cs: ucbCs, initialActionValueEstimates: ucbInitialActionValueEstimates)

    var ucbMeanAverageReward = [Double]()
    for actionTracker in ucbActionTrackers {
        actionTracker.computeMeanAverageReward()
        ucbMeanAverageReward.append(actionTracker.meanAverageReward)
    }

    // gradient bandit
    var gradActionTrackers = [ActionTracker]()
    let alphas = [1.0/32.0, 1.0/16.0, 1.0/8.0, 1.0/4.0, 1.0/2.0, 1.0, 2.0, 4.0]
    for _ in alphas {
        gradActionTrackers.append(ActionTracker())
    }
    let withBaselineFlags = Array(repeating: true, count: alphas.count)

    gradientBanditSimulateAll(actionTrackers: gradActionTrackers, alphas: alphas,
                              withBaselineFlags: withBaselineFlags, mean: 0)

    var gradMeanAverageReward = [Double]()
    for actionTracker in gradActionTrackers {
        actionTracker.computeMeanAverageReward()
        gradMeanAverageReward.append(actionTracker.meanAverageReward)
    }

    // plot everything

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)

    lineGraph.addSeries(optInitialActionValueEstimates.map{ log2($0) }, greedyOptimMeanAverageReward, label: "greedy with opt. init, step=0.1", color: Color.blue)
    lineGraph.addSeries(epsilons.map{ log2($0) }, epsilonGreedyMeanAverageReward, label: "eps-greedy", color: Color.orange)
    lineGraph.addSeries(ucbCs.map{ log2($0) }, ucbMeanAverageReward, label: "UCB", color: Color.green)
    lineGraph.addSeries(alphas.map{ log2($0) }, gradMeanAverageReward, label: "gradient bandit", color: Color.red)
    lineGraph.plotLabel.xLabel = "log2( eps alpha c Q_0 )"
    lineGraph.plotLabel.yLabel = "Average reward over first 1000 steps"
    try? lineGraph.drawGraphAndOutput(fileName: "output/Chapter2/Fig_2.6", renderer: aggRenderer)
}

make_figure_2_2()
make_figure_2_3()
make_figure_2_4()
make_figure_2_5()
make_figure_2_6()
