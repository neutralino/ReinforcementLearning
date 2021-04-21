import Foundation
import SwiftPlot
import AGGRenderer

func initializePolicy() -> [State: Bool] {
    // the default policy sticks if the player's sum is 20 or 21, else hits
    var policy = [State: Bool]()
    for isum in 12...30 {
        for idealer in 1...10 {
            if isum < 20 {
                policy[State(isum, idealer, true)] = true
                policy[State(isum, idealer, false)] = true
            } else {
                policy[State(isum, idealer, true)] = false
                policy[State(isum, idealer, false)] = false
            }
        }
    }
    return policy
}

func first_visit_MC_episode(v: inout [State: Double], returns: inout [State: [Int]]) {
    let blackJack = BlackJack()
    blackJack.startGame()

    let policy = initializePolicy()
    blackJack.runPlayerPolicy(policy: policy)
    blackJack.runDealerPolicy()

    // since all rewards within the game are 0, the total return is
    // just the final reward.
    let G = blackJack.computeReward()

    for (index, state) in blackJack.sequence.reversed().enumerated() {
        let originalIndex = blackJack.sequence.count - index - 1
        let precedingStates = blackJack.sequence[0..<originalIndex]

        // first-visit
        if !precedingStates.contains(state) {
            if returns.keys.contains(state) {
                returns[state]!.append(G)
            } else {
                returns[state] = [G]
            }
            v[state] = Double(returns[state]!.reduce(0, +)) / Double(returns[state]!.count)
        }
    }
}

func run_mc_policy_evaluation(n: Int) -> [State: Double] {
    var v = [State: Double]()
    var returns = [State: [Int]]()

    for _ in 0..<n {
        first_visit_MC_episode(v: &v, returns: &returns)
    }
    return v
}

func fill_figure_5_1_subplot(values: [(Int, Int, Double)]) -> Heatmap<[[Double]]> {
    var array = Array(repeating: Array(repeating: 0.0, count: 10), count: 10)

    for v in values {
        if v.0 >= 1 && v.1 >= 12 {
            array[v.1 - 12][v.0 - 1] = v.2
        }
    }
    let heatmap = Heatmap<[[Double]]>(array)
    return heatmap
}

func make_figure_5_1() {
    print("Generating Figure 5.1")
    let v10K = run_mc_policy_evaluation(n: 10000)
    let v500K = run_mc_policy_evaluation(n: 500000)

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 2, columns: 2))

    // filtering out sum >21 for plotting purposes
    // (hist2d bin plotting include the last bin edge)
    let v10KUseableAce = v10K.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map { ($0.key.dealer, $0.key.sum, $0.value) }
    let v10KNoUseableAce = v10K.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map { ($0.key.dealer, $0.key.sum, $0.value) }
    let v500KUseableAce = v500K.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map { ($0.key.dealer, $0.key.sum, $0.value) }
    let v500KNoUseableAce = v500K.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map { ($0.key.dealer, $0.key.sum, $0.value) }

    var plot = fill_figure_5_1_subplot(values: v10KUseableAce)
    plot.plotLabel.xLabel = "Dealer showing (minus 1)"
    plot.plotLabel.yLabel = "(No Useable Ace) Player sum (minus 12)"
    var plots = [plot]

    plot = fill_figure_5_1_subplot(values: v500KUseableAce)
    plot.plotLabel.xLabel = "Dealer showing (minus 1)"
    plots.append(plot)

    plot = fill_figure_5_1_subplot(values: v10KNoUseableAce)
    plot.plotTitle.title = "After 10,000 episodes"
    plot.plotLabel.yLabel = "(Useable Ace) Player sum (minus 12)"
    plots.append(plot)

    plot = fill_figure_5_1_subplot(values: v500KNoUseableAce)
    plot.plotTitle.title = "After 500,000 episodes"
    plots.append(plot)

    subplot.plots = plots
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter5/Fig_5.1", renderer: aggRenderer)

}

func first_visit_MC_q_episode(Q: inout [StateAction: Double],
                              returns: inout [StateAction: [Int]],
                              policy: inout [State: Bool]) {
    let blackJack = BlackJack()

    // randomly initialize game state and action
    blackJack.startRandomStateGame()
    let action0 = Bool.random() // 1 for 'hit', 0 for 'stick'
    blackJack.initializeStateActionSequence(action: action0)
    if action0 {
        blackJack.runPlayerPolicy(policy: policy)
    }
    blackJack.runDealerPolicy()

    // since all rewards within the game are 0, the total return is
    // just the final reward.
    let G = blackJack.computeReward()
    for (index, stateAction) in blackJack.stateActionSequence.reversed().enumerated() {
        let originalIndex = blackJack.stateActionSequence.count - index - 1
        let precedingStateActions = blackJack.stateActionSequence[0..<originalIndex]

        // first-visit
        if !precedingStateActions.contains(stateAction) {
            if returns.keys.contains(stateAction) {
                returns[stateAction]!.append(G)
            } else {
                returns[stateAction] = [G]
            }
            Q[stateAction] = Double(returns[stateAction]!.reduce(0, +)) / Double(returns[stateAction]!.count)

            let maxQ = Q.filter{ $0.key.state == stateAction.state }.max{ a, b in a.value < b.value }
            let bestAction = maxQ!.0.action
            policy[stateAction.state] = bestAction
        }
    }
}

func run_mc_es(n: Int) -> ([State: Double], [State: Bool]) {
    var Q = [StateAction: Double]()
    var returns = [StateAction: [Int]]()
    var policy = initializePolicy()

    // initialize Q values to zero
    for isum in 12...21 {
        for idealer in 1...10 {
            Q[StateAction(State(isum, idealer, true), false)] = 0.0
            Q[StateAction(State(isum, idealer, true), true)] = 0.0
            Q[StateAction(State(isum, idealer, false), false)] = 0.0
            Q[StateAction(State(isum, idealer, false), true)] = 0.0
        }
    }

    for _ in 0..<n {
        first_visit_MC_q_episode(Q: &Q, returns: &returns, policy: &policy)
    }

    // derive state-value function from action-value function
    var v = [State: Double]()
    for (stateAction, value) in Q {
        if v.keys.contains(stateAction.state) {
            v[stateAction.state] = max(v[stateAction.state]!, value)
        } else {
            v[stateAction.state] = value
        }
    }
    return (v, policy)
}

func make_figure_5_2() {
    print("Generating Figure 5.2")
    let vAndPolicy = run_mc_es(n: 1000000)
    let v = vAndPolicy.0
    let pi = vAndPolicy.1

    let aggRenderer: AGGRenderer = AGGRenderer()
    var subplot = SubPlot(layout: .grid(rows: 2, columns: 2))

    let vUseableAce = v.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let vNoUseableAce = v.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }

    let piUseableAce = pi.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value == true ? 1.0 : 0.0) }
    let piNoUseableAce = pi.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value == true ? 1.0 : 0.0) }

    var plot = fill_figure_5_1_subplot(values: piNoUseableAce)
    plot.plotLabel.xLabel = "Dealer showing (minus 1)"
    plot.plotLabel.yLabel = "(No Useable Ace) Player sum (minus 12)"
    var plots = [plot]

    plot = fill_figure_5_1_subplot(values: vNoUseableAce)
    plot.plotLabel.xLabel = "Dealer showing (minus 1)"
    plots.append(plot)

    plot = fill_figure_5_1_subplot(values: piUseableAce)
    plot.plotTitle.title = "pi_*"
    plot.plotLabel.yLabel = "(Useable Ace) Player sum (minus 12)"
    plots.append(plot)

    plot = fill_figure_5_1_subplot(values: vUseableAce)
    plot.plotTitle.title = "v_*"
    plots.append(plot)

    subplot.plots = plots
    try? subplot.drawGraphAndOutput(fileName: "Output/Chapter5/Fig_5.2", renderer: aggRenderer)

}

func make_figure_5_3() {
    print("Generating Figure 5.3")

    let nRuns = 100
    let nEpisodes = 10000

    let trueV = -0.27726
    var allVs = [[Double]]()
    var allWeightedVs = [[Double]]()
    var MSEs = Array(repeating: 0.0, count: nEpisodes)
    var MSEsWeightedVs = Array(repeating: 0.0, count: nEpisodes)

    for iRun in 0..<nRuns {
        if iRun % 10 == 0 {
            print("Run \(iRun)")
        }
        var rhos = [Double]()
        var Gs = [Double]()
        var Vs = [Double]()
        var weightedVs = [Double]()
        for _ in 0..<nEpisodes {
            let blackJack = BlackJack()

            // we want to evaluate the value of this state under the target policy
            blackJack.playerCards = [Card(1, Suit.spade), Card(2, Suit.spade)]
            blackJack.dealerCards = [Card(2, Suit.spade)]
            blackJack.dealerHit()
            blackJack.updateCurrentState()

            let targetPolicy = initializePolicy()

            blackJack.runPlayerRandomPolicy()
            blackJack.runDealerPolicy()

            let G = Double(blackJack.computeReward())

            // compute rho of this trajectory
            let denominator = pow(0.5, Double(blackJack.stateActionSequence.count))
            let numerator = blackJack.stateActionSequence.map { (targetPolicy[$0.state] == $0.action) ? 1.0 : 0.0 }.reduce(1, *)
            let rho = numerator / denominator

            Gs.append(G)
            rhos.append(rho)

            var vNumerator = 0.0
            for (p, g) in zip(rhos, Gs) {
                vNumerator += p * g
            }
            let V = vNumerator / Double(Gs.count)
            Vs.append(V)

            let weightedV = (rhos.reduce(0, +) == 0) ? 0.0 : vNumerator / rhos.reduce(0, +)
            weightedVs.append(weightedV)
        }
        allVs.append(Vs)
        allWeightedVs.append(weightedVs)
    }

    // compute MSEs
    for iEpisode in 0..<nEpisodes {
        for jRun in 0..<nRuns {
            MSEs[iEpisode] += pow(trueV - allVs[jRun][iEpisode], 2.0)
            MSEsWeightedVs[iEpisode] += pow(trueV - allWeightedVs[jRun][iEpisode], 2.0)
        }
        MSEs[iEpisode] /= Double(nRuns)
        MSEsWeightedVs[iEpisode] /= Double(nRuns)
    }

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>()

    let x: [Double] = Array(1...nEpisodes).map{ log10(Double($0)) }

    // there's some scaling bug in SwiftPlot's LineGraph that is preventing MSEs
    // from being drawn.  See https://github.com/KarthikRIyer/swiftplot/issues/126.
    // Commenting out for now.
    // lineGraph.addSeries(x,
    //                     MSEs,
    //                     label: "Ordinary importance sampling", color: Color.blue)
    lineGraph.addSeries(x,
                        MSEsWeightedVs,
                        label: "Weighted importance sampling", color: Color.orange)

    lineGraph.plotLabel.xLabel = "log10( Episodes )"
    lineGraph.plotLabel.yLabel = "Mean square error (average over 100 runs)"
    try? lineGraph.drawGraphAndOutput(fileName: "output/Chapter5/Fig_5.3", renderer: aggRenderer)
}

func make_figure_5_4() {
    print("Generating Figure 5.4")
    let nRuns = 10
    let nEpisodes = 100000

    let aggRenderer: AGGRenderer = AGGRenderer()
    var lineGraph = LineGraph<Double, Double>()
    let colorWheel = [Color.purple, Color.lightBlue, Color.blue, Color.darkBlue,
                      Color.green, Color.darkGreen, Color.yellow, Color.gold,
                      Color.orange, Color.red]

    for iRun in 0..<nRuns {
        print("Run #\(iRun)")
        var rhos = [Double]()
        var Gs = [Double]()
        var Vs = [Double]()

        for _ in 0..<nEpisodes {
            let infVarMDP = InfiniteVariance()
            var terminated = false
            var G = 0.0
            while !terminated {
                let stateAndReward = infVarMDP.getNextStateAndRewardBehaviorPolicy()
                terminated = stateAndReward.0
                G += Double(stateAndReward.1)
            }
            Gs.append(G)

            let denominator = pow(0.5, Double(infVarMDP.stateActionSequence.count))
            // target policy always chooses left
            let numerator = infVarMDP.stateActionSequence.map { $0.action == IVAction.left ? 1.0 : 0.0 }.reduce(1, *)
            let rho = numerator / denominator
            rhos.append(rho)

            var vNumerator = 0.0
            for (p, g) in zip(rhos, Gs) {
                vNumerator += p * g
            }
            let V = vNumerator / Double(Gs.count)
            Vs.append(V)
        }

        let x: [Double] = Array(1...nEpisodes).map{ log10(Double($0)) }
        lineGraph.addSeries(x,
                            Vs,
                            label: "", color: colorWheel[iRun])

    }

    lineGraph.plotLabel.xLabel = "log10( Episodes )"
    lineGraph.plotLabel.yLabel = "MC estimate of v_pi(s) with ordinary importance sampling\n(10 runs)"

    // should ideally limit the y range from 0 to 4, but SwiftPlot doesn't
    // seem to support range definition at the moment.
    try? lineGraph.drawGraphAndOutput(fileName: "output/Chapter5/Fig_5.4", renderer: aggRenderer)

}

//make_figure_5_1()
//make_figure_5_2()
//make_figure_5_3()
make_figure_5_4()
