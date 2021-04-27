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
    // Bug in SwiftPlot's LineGraph preventing lineGraph2 from being plotted without scaling y values to be less than 2
    lineGraph2.addSeries(xTrajectory.map { Double($0) }, yTrajectory.map { Double($0) / 10 }, label: "")
    lineGraph2.plotTitle.title = "N steps for final Q: \(xTrajectory.count-1)"
    lineGraph2.plotLabel.xLabel = "Grid point"
    lineGraph2.plotLabel.yLabel = "Grid point / 10"

    subplot.plots = [lineGraph, lineGraph2]
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

func runSarsa(cliffWorld: CliffWorld, alpha: Double, epsilon: Double, maxEpisodes: Int, initialS: Point, terminalS: Point) -> ([[Double]], [Int]) {
    var Q = Array(repeating: Array(repeating: 0.0, count: 4), count: cliffWorld.nX * cliffWorld.nY)
    var nCompletedEpisodes = 0
    var sumRewardTracker = [Int]()

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
    return (Q, sumRewardTracker)
}

func runQLearning(cliffWorld: CliffWorld, alpha: Double, epsilon: Double, maxEpisodes: Int, initialS: Point, terminalS: Point, expectedSarsa: Bool = false) -> ([[Double]], [Int]) {
    var Q = Array(repeating: Array(repeating: 0.0, count: 4), count: cliffWorld.nX * cliffWorld.nY)
    var sumRewardTracker = [Int]()

    var nCompletedEpisodes = 0
    while nCompletedEpisodes < maxEpisodes {
        var S = initialS
        var sumReward = 0
        while S != terminalS {
            let A = chooseActionFromQ(Q: Q, S: S, epsilon: epsilon, gridWorld: cliffWorld)
            let nextSAndR = cliffWorld.move(S, A)
            let nextS = nextSAndR.0
            let R = nextSAndR.1
            sumReward += R

            let currentQ = Q[cliffWorld.gridToInt(S)][A.rawValue]
            var nextQ = 0.0

            if nextS != terminalS {
                let nextSInt = cliffWorld.gridToInt(nextS)
                if expectedSarsa {
                    let maxQ = Q[nextSInt].max()
                    let maxQIndices = Q[nextSInt].indices.filter { Q[nextSInt][$0] == maxQ }

                    for iAction in 0...3 {
                        var actionProb = 0.0
                        if maxQIndices.contains(iAction) {
                            actionProb = (1.0 - epsilon) / Double(maxQIndices.count) + epsilon / 4.0
                        } else {
                            actionProb = epsilon / 4.0
                        }
                        nextQ += actionProb * Q[nextSInt][iAction]
                    }
                } else {
                    nextQ = Q[nextSInt].max()!
                }
            }

            Q[cliffWorld.gridToInt(S)][A.rawValue] += alpha * (Double(R) + nextQ - currentQ)

            S = nextS
        }
        nCompletedEpisodes += 1
        sumRewardTracker.append(sumReward)
    }
    return (Q, sumRewardTracker)
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

    var allSumRewardsSarsa = [[Int]]()
    var allSumRewardsQLearning = [[Int]]()

    var finalRunSarsaQ = Array(repeating: Array(repeating: 0.0, count: 4), count: nX * nY)
    var finalRunQLearningQ = Array(repeating: Array(repeating: 0.0, count: 4), count: nX * nY)

    for _ in 0..<nRuns {

        let sarsaQAndTracker = runSarsa(cliffWorld: cliffWorld, alpha: alpha,
                                        epsilon: epsilon, maxEpisodes: maxEpisodes,
                                        initialS: initialS, terminalS: terminalS)
        finalRunSarsaQ = sarsaQAndTracker.0
        allSumRewardsSarsa.append(sarsaQAndTracker.1)

        let qLearningQAndTracker = runQLearning(cliffWorld: cliffWorld,
                                                alpha: alpha, epsilon: epsilon,
                                                maxEpisodes: maxEpisodes,
                                                initialS: initialS, terminalS: terminalS)
        finalRunQLearningQ = qLearningQAndTracker.0
        allSumRewardsQLearning.append(qLearningQAndTracker.1)

    }
    let averageRewardsRunsSarsa = computeMean(allRewardsRuns: allSumRewardsSarsa)
    let averageRewardsRunsQLearning = computeMean(allRewardsRuns: allSumRewardsQLearning)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 1, columns: 2))
    var lineGraph = LineGraph<Double, Double>()

    lineGraph.addSeries(Array(0..<maxEpisodes).map { Double($0) }, averageRewardsRunsSarsa, label: "Sarsa", color: Color.blue)
    lineGraph.addSeries(Array(0..<maxEpisodes).map { Double($0) }, averageRewardsRunsQLearning, label: "Q-learning", color: Color.red)
    lineGraph.plotLabel.xLabel = "(Average) sum of rewards during episode"
    lineGraph.plotLabel.yLabel = "Episodes"
    lineGraph.plotTitle.title = "Cliff Walking"

    let (xTrajectory, yTrajectory) = getGreedyTrajectory(initialS: initialS, terminalS: terminalS,
                                                         Q: finalRunSarsaQ, gridWorld: cliffWorld)

    let (xTrajectory2, yTrajectory2) = getGreedyTrajectory(initialS: initialS, terminalS: terminalS,
                                                           Q: finalRunQLearningQ, gridWorld: cliffWorld)

    var lineGraph2 = LineGraph<Double, Double>(enablePrimaryAxisGrid: true)
    // Bug in SwiftPlot's LineGraph preventing lineGraph2 from being plotted without scaling y values to be less than 2
    lineGraph2.addSeries(xTrajectory.map { Double($0) }, yTrajectory.map { Double($0) / 100 }, label: "Sarsa", color: Color.blue)
    lineGraph2.addSeries(xTrajectory2.map { Double($0) }, yTrajectory2.map { Double($0) / 100 }, label: "Q-learning", color: Color.red)
    lineGraph2.plotLabel.xLabel = "grid point"
    lineGraph2.plotLabel.yLabel = "grid point / 100"
    lineGraph2.plotTitle.title = "Sample final greedy policy"
    subplot.plots = [lineGraph, lineGraph2]
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter6/Example_6.6", renderer: aggRenderer)
}

func make_figure_6_3() {
    print("Generating Figure 6.3")

    let nX = 12
    let nY = 4
    let cliffWorld = CliffWorld(nX, nY)

    let alphas = stride(from: 0.1, to: 1.05, by: 0.05)
    let epsilon = 0.1
    let terminalS = Point(11, 0)
    let initialS = Point(0, 0)

    func computeAverageSumRewardsPerEpisode(sumRewardsInEpisode: [[Int]], numEpisodes: Int, numRuns: Int) -> Double {
        let sumRewardsPerEpisode = sumRewardsInEpisode.map { Double($0.reduce(0, +)) / Double(numEpisodes) }
        return Double(sumRewardsPerEpisode.reduce(0, +)) / Double(numRuns)
    }

    // interim performance
    // var nRuns = 50000
    var nRuns = 5000
    var maxEpisodes = 100

    var avgSumRewardsPerEpisodePerAlphaSarsa = [Double]()
    var avgSumRewardsPerEpisodePerAlphaQLearning = [Double]()
    var avgSumRewardsPerEpisodePerAlphaExpSarsa = [Double]()

    for alpha in alphas {
        print(alpha)
        var allSumRewardsSarsa = [[Int]]()
        var allSumRewardsQLearning = [[Int]]()
        var allSumRewardsExpSarsa = [[Int]]()

        for _ in 0..<nRuns {
            let sarsaQAndTracker = runSarsa(cliffWorld: cliffWorld, alpha: alpha,
                                            epsilon: epsilon, maxEpisodes: maxEpisodes,
                                            initialS: initialS, terminalS: terminalS)
            allSumRewardsSarsa.append(sarsaQAndTracker.1)

            let qLearningQAndTracker = runQLearning(cliffWorld: cliffWorld,
                                                    alpha: alpha, epsilon: epsilon,
                                                    maxEpisodes: maxEpisodes,
                                                    initialS: initialS, terminalS: terminalS)
            allSumRewardsQLearning.append(qLearningQAndTracker.1)

            let expSarsaQAndTracker = runQLearning(cliffWorld: cliffWorld,
                                                    alpha: alpha, epsilon: epsilon,
                                                    maxEpisodes: maxEpisodes,
                                                    initialS: initialS, terminalS: terminalS, expectedSarsa: true)
            allSumRewardsExpSarsa.append(expSarsaQAndTracker.1)

        }

        let avgSumRewardsPerEpisodeSarsa = computeAverageSumRewardsPerEpisode(sumRewardsInEpisode: allSumRewardsSarsa, numEpisodes: maxEpisodes, numRuns: nRuns)
        avgSumRewardsPerEpisodePerAlphaSarsa.append(avgSumRewardsPerEpisodeSarsa)

        let avgSumRewardsPerEpisodeQLearning = computeAverageSumRewardsPerEpisode(sumRewardsInEpisode: allSumRewardsQLearning, numEpisodes: maxEpisodes, numRuns: nRuns)
        avgSumRewardsPerEpisodePerAlphaQLearning.append(avgSumRewardsPerEpisodeQLearning)

        let avgSumRewardsPerEpisodeExpSarsa = computeAverageSumRewardsPerEpisode(sumRewardsInEpisode: allSumRewardsExpSarsa, numEpisodes: maxEpisodes, numRuns: nRuns)
        avgSumRewardsPerEpisodePerAlphaExpSarsa.append(avgSumRewardsPerEpisodeExpSarsa)
    }

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>()
    lineGraph.addSeries(alphas.map { Double($0) }, avgSumRewardsPerEpisodePerAlphaSarsa.map { $0 / 1000.0 }, label: "Sarsa (Interim)", color: Color.blue)
    lineGraph.addSeries(alphas.map { Double($0) }, avgSumRewardsPerEpisodePerAlphaQLearning.map { $0 / 1000.0 }, label: "Q-learning (Interim)", color: Color.gray)
    lineGraph.addSeries(alphas.map { Double($0) }, avgSumRewardsPerEpisodePerAlphaExpSarsa.map { $0 / 1000.0 }, label: "Expected (Interim)", color: Color.red)

    // asymptotic performance
    nRuns = 10
    maxEpisodes = 100000

    avgSumRewardsPerEpisodePerAlphaSarsa = [Double]()
    avgSumRewardsPerEpisodePerAlphaQLearning = [Double]()
    avgSumRewardsPerEpisodePerAlphaExpSarsa = [Double]()

    for alpha in alphas {
        print(alpha)
        var allSumRewardsSarsa = [[Int]]()
        var allSumRewardsQLearning = [[Int]]()
        var allSumRewardsExpSarsa = [[Int]]()

        for _ in 0..<nRuns {
            let sarsaQAndTracker = runSarsa(cliffWorld: cliffWorld, alpha: alpha,
                                            epsilon: epsilon, maxEpisodes: maxEpisodes,
                                            initialS: initialS, terminalS: terminalS)
            allSumRewardsSarsa.append(sarsaQAndTracker.1)

            let qLearningQAndTracker = runQLearning(cliffWorld: cliffWorld,
                                                    alpha: alpha, epsilon: epsilon,
                                                    maxEpisodes: maxEpisodes,
                                                    initialS: initialS, terminalS: terminalS)
            allSumRewardsQLearning.append(qLearningQAndTracker.1)

            let expSarsaQAndTracker = runQLearning(cliffWorld: cliffWorld,
                                                    alpha: alpha, epsilon: epsilon,
                                                    maxEpisodes: maxEpisodes,
                                                    initialS: initialS, terminalS: terminalS, expectedSarsa: true)
            allSumRewardsExpSarsa.append(expSarsaQAndTracker.1)

        }

        let avgSumRewardsPerEpisodeSarsa = computeAverageSumRewardsPerEpisode(sumRewardsInEpisode: allSumRewardsSarsa, numEpisodes: maxEpisodes, numRuns: nRuns)
        avgSumRewardsPerEpisodePerAlphaSarsa.append(avgSumRewardsPerEpisodeSarsa)

        let avgSumRewardsPerEpisodeQLearning = computeAverageSumRewardsPerEpisode(sumRewardsInEpisode: allSumRewardsQLearning, numEpisodes: maxEpisodes, numRuns: nRuns)
        avgSumRewardsPerEpisodePerAlphaQLearning.append(avgSumRewardsPerEpisodeQLearning)

        let avgSumRewardsPerEpisodeExpSarsa = computeAverageSumRewardsPerEpisode(sumRewardsInEpisode: allSumRewardsExpSarsa, numEpisodes: maxEpisodes, numRuns: nRuns)
        avgSumRewardsPerEpisodePerAlphaExpSarsa.append(avgSumRewardsPerEpisodeExpSarsa)
    }

    // Bug in SwiftPlot's LineGraph preventing lineGraph from being plotted without scaling y values to be less than 2
    lineGraph.addSeries(alphas.map { Double($0) }, avgSumRewardsPerEpisodePerAlphaSarsa.map { $0 / 1000.0 }, label: "Sarsa (Asymptotic)", color: Color.darkBlue)
    lineGraph.addSeries(alphas.map { Double($0) }, avgSumRewardsPerEpisodePerAlphaQLearning.map { $0 / 1000.0 }, label: "Q-learning (Asymptotic)", color: Color.darkGray)
    lineGraph.addSeries(alphas.map { Double($0) }, avgSumRewardsPerEpisodePerAlphaExpSarsa.map { $0 / 1000.0 }, label: "Expected (Asymptotic)", color: Color.darkRed)

    lineGraph.plotLabel.xLabel = "alpha"
    lineGraph.plotLabel.yLabel = "Average sum of rewards per episode / 1000"
    lineGraph.plotTitle.title = "Cliff Walking"
    try? lineGraph.drawGraphAndOutput(fileName: "Output/Chapter6/Fig_6.3", renderer: aggRenderer)
}

make_example_6_2()
make_figure_6_2()
make_example_6_5()
make_example_6_6()
make_figure_6_3()
