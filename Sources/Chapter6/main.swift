import Foundation
import SwiftPlot
import AGGRenderer

let stateToIndex = [State.A: 0, State.B: 1, State.C: 2, State.D: 3, State.E: 4]
let trueV = Array(1...5).map { Double($0) / 6.0 }

func run_td0_episode(alpha: Double, V: inout [Double]) {
    let randomWalk = RandomWalk()
    randomWalk.generateStateRewardSequence()

    let seq = randomWalk.stateRewardSequence
    for (i, stateReward) in seq.enumerated() {
        let nextVS = (i < seq.count - 1) ? V[stateToIndex[seq[i+1].state]!] : 0.0
        let delta = stateReward.reward + nextVS - V[stateToIndex[stateReward.state]!]
        V[stateToIndex[stateReward.state]!] += alpha * delta
    }
}

// every-visit MC from Eqn 6.1
func run_mc_episode(alpha: Double, V: inout [Double]) {
    let randomWalk = RandomWalk()
    randomWalk.generateStateRewardSequence()

    let seq = randomWalk.stateRewardSequence
    // In this problem, the return is just the final reward.
    let G = seq.last!.reward
    for stateReward in seq {
        let stateIndex = stateToIndex[stateReward.state]!
        let delta = G - V[stateIndex]
        V[stateIndex] += alpha * delta
    }
}

func run_batch_td0_update(episodes: [[StateReward]], alpha: Double, V: inout [Double]) {
    let minDelta = 1e-5
    var minUpdate = 1.0
    while minUpdate > minDelta {
        var updates = Array(repeating: 0.0, count: V.count)

        for seq in episodes {
            for (i, stateReward) in seq.enumerated() {
                let stateIndex = stateToIndex[stateReward.state]!
                let nextVS = (i < seq.count - 1) ? V[stateToIndex[seq[i+1].state]!] : 0.0
                let delta = stateReward.reward + nextVS - V[stateIndex]
                updates[stateIndex] += alpha * delta
            }
        }

        for (stateIndex, update) in updates.enumerated() {
            V[stateIndex] += update
        }
        minUpdate = updates.max()!
    }
}

func run_batch_mc_update(episodes: [[StateReward]], alpha: Double, V: inout [Double]) {
    let minDelta = 1e-5
    var minUpdate = 1.0
    while minUpdate > minDelta {
        var updates = Array(repeating: 0.0, count: V.count)

        for seq in episodes {
            let G = seq.last!.reward
            for stateReward in seq {
                let stateIndex = stateToIndex[stateReward.state]!
                let delta = G - V[stateIndex]
                updates[stateIndex] += alpha * delta
            }
        }

        for (stateIndex, update) in updates.enumerated() {
            V[stateIndex] += update
        }
        minUpdate = updates.max()!
    }
}

func fill_value_estimate_subplot(initialV: [Double]) -> LineGraph<Double, Double> {
    let l = Array(1...5).map { Double($0) }
    var V = initialV

    var lineGraph = LineGraph<Double, Double>()
    lineGraph.addSeries(l, trueV, label: "True value", color: Color.blue)
    lineGraph.addSeries(l, V, label: "0 Episodes", color: Color.orange)

    let alpha = 0.1

    let colorWheel = [Color.green, Color.red, Color.purple]
    var colorWheelIterator = colorWheel.makeIterator()
    for nEpisodes in [1, 10, 100] {
        for _ in 0..<nEpisodes {
            run_td0_episode(alpha: alpha, V: &V)
        }
        lineGraph.addSeries(l, V, label: "\(nEpisodes) Episodes", color: colorWheelIterator.next()!)
    }

    lineGraph.plotLabel.xLabel = "State"
    lineGraph.plotTitle.title = "Estimated value"
    return lineGraph
}

func generateValueEstimates(initialV: [Double], alpha: Double, nRuns: Int, nEpisodes: Int, TD: Bool) -> [[[Double]]] {
    var allVsRuns = [[[Double]]]()

    for _ in 0..<nRuns {
        var V = initialV
        var allVs = [[Double]]()
        for _ in 0..<nEpisodes {
            if TD {
                run_td0_episode(alpha: alpha, V: &V)
            } else {
                run_mc_episode(alpha: alpha, V: &V)
            }
            allVs.append(V)
        }
        allVsRuns.append(allVs)
    }
    return allVsRuns
}

func computeRMSE(allVsRuns: [[[Double]]]) -> [Double] {
    // compute RMSEs
    let nRuns = allVsRuns.count
    let nEpisodes = allVsRuns[0].count
    var MSEs = Array(repeating: 0.0, count: nEpisodes)

    for iEpisode in 0..<nEpisodes {
        for jRun in 0..<nRuns {
            // compute first the MSE averaged over all state
            var mse = 0.0
            for kState in 0..<trueV.count {
                mse += pow(trueV[kState] - allVsRuns[jRun][iEpisode][kState], 2.0)
            }
            mse /= Double(trueV.count)
            MSEs[iEpisode] += mse
        }
        MSEs[iEpisode] /= Double(nRuns)
    }

    let RMSEs = MSEs.map { sqrt($0) }
    return RMSEs
}

func fill_learning_curve_subplot(initialV: [Double]) -> LineGraph<Double, Double> {
    let nRuns = 100
    let nEpisodes = 100

    var lineGraph = LineGraph<Double, Double>()
    let colorWheel = [Color.blue, Color.orange, Color.green, Color.red, Color.purple, Color.brown, Color.pink]
    var colorWheelIterator = colorWheel.makeIterator()

    for alpha in [0.05, 0.1, 0.15] {
        let allVsRuns = generateValueEstimates(initialV: initialV, alpha: alpha,
                                               nRuns: nRuns, nEpisodes: nEpisodes,
                                               TD: true)
        let RMSEs = computeRMSE(allVsRuns: allVsRuns)

        lineGraph.addSeries(Array(0..<nEpisodes).map { Double($0) }, RMSEs, label: "TD: alpha=\(alpha)", color: colorWheelIterator.next()!)
    }

    for alpha in [0.01, 0.02, 0.03, 0.04] {
        let allVsRuns = generateValueEstimates(initialV: initialV, alpha: alpha,
                                               nRuns: nRuns, nEpisodes: nEpisodes,
                                               TD: false)
        let RMSEs = computeRMSE(allVsRuns: allVsRuns)

        lineGraph.addSeries(Array(0..<nEpisodes).map { Double($0) }, RMSEs, label: "MC: alpha=\(alpha)", color: colorWheelIterator.next()!)

    }

    lineGraph.plotLabel.xLabel = "Episodes"
    lineGraph.plotTitle.title = "RMS error, averaged over states"
    return lineGraph

}

func generateBatchedUpdateEstimates(initialV: [Double], alpha: Double, nRuns: Int, nEpisodes: Int, TD: Bool) -> [[[Double]]] {
    var allVsRuns = [[[Double]]]()

    for iRun in 0..<nRuns {
        if iRun % 10 == 0 {
            print("Run #\(iRun+1)")
        }
        var allVs = [[Double]]()
        var episodes = [[StateReward]]()
        for _ in 0..<nEpisodes {
            var V = initialV
            let randomWalk = RandomWalk()
            randomWalk.generateStateRewardSequence()
            let seq = randomWalk.stateRewardSequence
            episodes.append(seq)

            if TD {
                run_batch_td0_update(episodes: episodes, alpha: alpha, V: &V)
            } else {
                run_batch_mc_update(episodes: episodes, alpha: alpha, V: &V)
            }
            allVs.append(V)
        }
        allVsRuns.append(allVs)
    }
    return allVsRuns
}

func make_example_6_2() {
    print("Generating Example 6.2")

    // initial value function
    let V = Array(repeating: 0.5, count: 5)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 1, columns: 2))

    var plot = fill_value_estimate_subplot(initialV: V)
    var plots = [plot]

    plot = fill_learning_curve_subplot(initialV: V)
    plots.append(plot)

    subplot.plots = plots
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter6/Example_6.2", renderer: aggRenderer)
}

func make_figure_6_2() {
    print("Generating Figure 6.2")

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>()

    let nRuns = 100
    let nEpisodes = 100
    let alpha = 0.001

    // initial value function
    let V = Array(repeating: 0.5, count: 5)

    var allVsRuns = generateBatchedUpdateEstimates(initialV: V, alpha: alpha, nRuns: nRuns, nEpisodes: nEpisodes, TD: true)
    var RMSEs = computeRMSE(allVsRuns: allVsRuns)
    lineGraph.addSeries(Array(0..<nEpisodes).map { Double($0) }, RMSEs, label: "TD: alpha=\(alpha)", color: Color.blue)

    allVsRuns = generateBatchedUpdateEstimates(initialV: V, alpha: alpha, nRuns: nRuns, nEpisodes: nEpisodes, TD: false)
    RMSEs = computeRMSE(allVsRuns: allVsRuns)
    lineGraph.addSeries(Array(0..<nEpisodes).map { Double($0) }, RMSEs, label: "MC: alpha=\(alpha)", color: Color.red)

    lineGraph.plotLabel.xLabel = "Episodes"
    lineGraph.plotLabel.yLabel = "RMS error, averaged over states"
    lineGraph.plotTitle.title = "Batch Training"
    try? lineGraph.drawGraphAndOutput(fileName: "Output/Chapter6/Fig_6.2", renderer: aggRenderer)
}

func chooseActionFromQ(Q: [[Double]], S: Point, epsilon: Double, gridWorld: GridWorld) -> Action {
    // select the action a that maximizes Q(S, a)
    let SInt = gridWorld.gridToInt(S)
    let maxQ = Q[SInt].max()
    let maxQIndices = Q[SInt].indices.filter { Q[SInt][$0] == maxQ }

    // epsilon greedy
    let d = Double.random(in: 0...1)
    let chosenIndex = (d <= epsilon) ? Q[SInt].indices.randomElement()! : maxQIndices.randomElement()!
    return Action(rawValue: chosenIndex)!
}

func getGreedyTrajectory(initialS: Point, terminalS: Point, Q: [[Double]], gridWorld: GridWorld) -> ([Int], [Int]) {
    var xTrajectory = [Int]()
    var yTrajectory = [Int]()
    var S = initialS
    xTrajectory.append(S.x)
    yTrajectory.append(S.y)
    while S != terminalS {
        let A = chooseActionFromQ(Q: Q, S: S, epsilon: 0.0, gridWorld: gridWorld)
        S = gridWorld.move(S, A).0
        xTrajectory.append(S.x)
        yTrajectory.append(S.y)
    }
    return (xTrajectory, yTrajectory)
}

func make_example_6_5() {
    print("Generating Example 6.5")
    let nX = 10
    let nY = 7
    let gridWorld = GridWorld(nX, nY)

    var Q = Array(repeating: Array(repeating: 0.0, count: 4), count: nX * nY)
    let epsilon = 0.1
    let alpha = 0.5
    let terminalS = Point(7, 3)
    let initialS = Point(0, 3)
    var nStep = 0
    var nCompletedEpisodes = 0
    var episodeTracker = [Int]()
    while nCompletedEpisodes < 200 {
        var S = initialS
        var A = chooseActionFromQ(Q: Q, S: S, epsilon: epsilon, gridWorld: gridWorld)

        while S != terminalS {
            let nextSAndR = gridWorld.move(S, A)
            let nextS = nextSAndR.0
            let R = nextSAndR.1

            let nextA = chooseActionFromQ(Q: Q, S: nextS, epsilon: epsilon, gridWorld: gridWorld)
            let currentQ = Q[gridWorld.gridToInt(S)][A.rawValue]
            let nextQ = (nextS == terminalS) ? 0 : Q[gridWorld.gridToInt(nextS)][nextA.rawValue]
            Q[gridWorld.gridToInt(S)][A.rawValue] += alpha * (Double(R) + nextQ - currentQ)

            S = nextS
            A = nextA
            nStep += 1
            episodeTracker.append(nCompletedEpisodes)
        }
        nCompletedEpisodes += 1
    }

    let (xTrajectory, yTrajectory) = getGreedyTrajectory(initialS: initialS, terminalS: terminalS, Q: Q, gridWorld: gridWorld)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 1, columns: 2))
    var lineGraph = LineGraph<Double, Double>()
    lineGraph.addSeries(Array(0..<nStep).map { Double($0) }, episodeTracker.map { Double($0) }, label: "")
    lineGraph.plotLabel.xLabel = "Time steps"
    lineGraph.plotLabel.yLabel = "Episodes"
    lineGraph.plotTitle.title = "Windy Gridworld"

    var lineGraph2 = LineGraph<Double, Double>()
    lineGraph2.addSeries(xTrajectory.map { Double($0) }, yTrajectory.map { Double($0) }, label: "")
    lineGraph2.plotTitle.title = "N steps for final Q: \(xTrajectory.count-1)"

    // Bug in SwiftPlot's LineGraph preventing lineGraph2 from being plotted
    // Disable for now
    // subplot.plots = [lineGraph, lineGraph2]
    subplot.plots = [lineGraph, lineGraph]
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter6/Example_6.5", renderer: aggRenderer)
}

func computeMean(allRewardsRuns: [[Int]]) -> [Double] {
    let nRuns = allRewardsRuns.count
    let nEpisodes = allRewardsRuns[0].count
    var means = Array(repeating: 0.0, count: nEpisodes)

    for iEpisode in 0..<nEpisodes {
        for jRun in 0..<nRuns {
            means[iEpisode] += Double(allRewardsRuns[jRun][iEpisode])
        }
        means[iEpisode] /= Double(nRuns)
    }
    return means
}

func make_example_6_6() {
    print("Generating Example 6.6")

    let nX = 12
    let nY = 4
    let cliffWorld = CliffWorld(nX, nY)

    let nRuns = 1000
    let maxEpisodes = 500

    let epsilon = 0.1
    let alpha = 0.5
    let terminalS = Point(11, 0)
    let initialS = Point(0, 0)

    var allSumRewards = [[Int]]()
    var allSumRewards2 = [[Int]]()

    var finalRunSarsaQ = Array(repeating: Array(repeating: 0.0, count: 4), count: nX * nY)
    var finalRunQLearningQ = Array(repeating: Array(repeating: 0.0, count: 4), count: nX * nY)

    for _ in 0..<nRuns {
        // Q: Sarsa, Q2: Q-Learning
        var Q = Array(repeating: Array(repeating: 0.0, count: 4), count: nX * nY)
        var Q2 = Array(repeating: Array(repeating: 0.0, count: 4), count: nX * nY)

        var nCompletedEpisodes = 0
        var sumRewardTracker = [Int]()
        var sumRewardTracker2 = [Int]()

        // Sarsa
        while nCompletedEpisodes < maxEpisodes {
            var S = initialS
            var A = chooseActionFromQ(Q: Q, S: S, epsilon: epsilon, gridWorld: cliffWorld)
            var sumReward = 0
            while S != terminalS {
                let nextSAndR = cliffWorld.move(S, A)
                let nextS = nextSAndR.0
                let R = nextSAndR.1
                sumReward += R

                let nextA = chooseActionFromQ(Q: Q, S: nextS, epsilon: epsilon, gridWorld: cliffWorld)
                let currentQ = Q[cliffWorld.gridToInt(S)][A.rawValue]
                let nextQ = (nextS == terminalS) ? 0 : Q[cliffWorld.gridToInt(nextS)][nextA.rawValue]
                Q[cliffWorld.gridToInt(S)][A.rawValue] += alpha * (Double(R) + nextQ - currentQ)

                S = nextS
                A = nextA
            }
            nCompletedEpisodes += 1
            sumRewardTracker.append(sumReward)
        }
        allSumRewards.append(sumRewardTracker)

        finalRunSarsaQ = Q

        // Q-learning
        nCompletedEpisodes = 0
        while nCompletedEpisodes < maxEpisodes {
            var S = initialS
            var sumReward2 = 0
            while S != terminalS {
                let A = chooseActionFromQ(Q: Q2, S: S, epsilon: epsilon, gridWorld: cliffWorld)
                let nextSAndR = cliffWorld.move(S, A)
                let nextS = nextSAndR.0
                let R = nextSAndR.1
                sumReward2 += R

                let currentQ = Q2[cliffWorld.gridToInt(S)][A.rawValue]
                let nextQ = (nextS == terminalS) ? 0 : Q2[cliffWorld.gridToInt(nextS)].max()!
                Q2[cliffWorld.gridToInt(S)][A.rawValue] += alpha * (Double(R) + nextQ - currentQ)

                S = nextS
            }
            nCompletedEpisodes += 1
            sumRewardTracker2.append(sumReward2)
        }
        allSumRewards2.append(sumRewardTracker2)

        finalRunQLearningQ = Q2
    }
    let averageRewardsRuns = computeMean(allRewardsRuns: allSumRewards)
    let averageRewardsRuns2 = computeMean(allRewardsRuns: allSumRewards2)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 1, columns: 2))
    var lineGraph = LineGraph<Double, Double>()

    lineGraph.addSeries(Array(0..<maxEpisodes).map { Double($0) }, averageRewardsRuns, label: "Sarsa", color: Color.blue)
    lineGraph.addSeries(Array(0..<maxEpisodes).map { Double($0) }, averageRewardsRuns2, label: "Q-learning", color: Color.red)
    lineGraph.plotLabel.xLabel = "(Average) sum of rewards during episode"
    lineGraph.plotLabel.yLabel = "Episodes"
    lineGraph.plotTitle.title = "Cliff Walking"

    let (xTrajectory, yTrajectory) = getGreedyTrajectory(initialS: initialS, terminalS: terminalS,
                                                         Q: finalRunSarsaQ, gridWorld: cliffWorld)

    let (xTrajectory2, yTrajectory2) = getGreedyTrajectory(initialS: initialS, terminalS: terminalS,
                                                           Q: finalRunQLearningQ, gridWorld: cliffWorld)

    var lineGraph2 = LineGraph<Double, Double>()
    lineGraph2.addSeries(xTrajectory.map { Double($0) }, yTrajectory.map { Double($0) }, label: "Sarsa", color: Color.blue)
    lineGraph2.addSeries(xTrajectory2.map { Double($0) }, yTrajectory2.map { Double($0) }, label: "Q-learning", color: Color.red)
    lineGraph2.plotTitle.title = "Sample final greedy policy"

    // Bug in SwiftPlot's LineGraph preventing lineGraph2 from being plotted
    // Disable for now
    // subplot.plots = [lineGraph, lineGraph2]
    subplot.plots = [lineGraph, lineGraph]
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter6/Example_6.6", renderer: aggRenderer)
}

make_example_6_2()
make_figure_6_2()
make_example_6_5()
make_example_6_6()
