import Foundation
import Python

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")


func initializePolicy() -> [State: Bool] {
    // the default policy sticks if the player's sum is 20 or 21, else hits
    var policy = [State: Bool]()
    for isum in 12...21 {
        for idealer in 1...10 {
            if isum < 20 {
                policy[State(isum, idealer, true)] = true
                policy[State(isum, idealer, false)] = true
            }
            else {
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

        //first-visit
        if !precedingStates.contains(state) {
            if returns.keys.contains(state) {
                returns[state]!.append(G)
            }
            else {
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

func fill_figure_5_1_subplot(figAndAx: (PythonObject, PythonObject), subplotIndex: (Int, Int),
                             values: [(Int, Int, Double)]) {
    let fig = figAndAx.0
    let ax = figAndAx.1

    let dealerRange = np.arange(1,12, 1)
    let sumRange = np.arange(12,23,1)

    let valuesDealer = values.map{ $0.0 }
    let valuesSum = values.map{ $0.1 }
    let valuesValue = values.map{ $0.2 }

    let row = subplotIndex.0
    let col = subplotIndex.1
    let h = ax[row][col].hist2d(valuesDealer, valuesSum, weights: valuesValue, cmap: "YlGn",
                                bins: [dealerRange, sumRange], vmin: -1, vmax: 1)

    if row == 1 && col == 0 {
        ax[row][col].set_xlabel("Dealer showing")
        ax[row][col].set_ylabel("(No Useable Ace)\nPlayer sum")
    }
    if row == 0 && col == 0 {
        ax[row][col].set_title("After 10,000 episodes")
        ax[row][col].set_ylabel("(Useable Ace)\nPlayer sum")
    }
    if row == 0 && col == 1 {
        ax[row][col].set_title("After 500,000 episodes")
    }

    fig.colorbar(h[3], ax: ax[row][col])
}

func make_figure_5_1() {
    print("Generating Figure 5.1")
    let v10K = run_mc_policy_evaluation(n: 10000)
    let v500K = run_mc_policy_evaluation(n: 500000)

    let figAndAx = plt.subplots(nrows: 2, ncols: 2, figsize: [8, 6.4]).tuple2

    // filtering out sum >21 for plotting purposes
    // (hist2d bin plotting include the last bin edge)
    let v10KUseableAce = v10K.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let v10KNoUseableAce = v10K.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let v500KUseableAce = v500K.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let v500KNoUseableAce = v500K.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }

    fill_figure_5_1_subplot(figAndAx: figAndAx, subplotIndex: (0, 0), values: v10KUseableAce)
    fill_figure_5_1_subplot(figAndAx: figAndAx, subplotIndex: (1, 0), values: v10KNoUseableAce)
    fill_figure_5_1_subplot(figAndAx: figAndAx, subplotIndex: (0, 1), values: v500KUseableAce)
    fill_figure_5_1_subplot(figAndAx: figAndAx, subplotIndex: (1, 1), values: v500KNoUseableAce)

    plt.savefig("Fig_5.1.png")
}


func first_visit_MC_q_episode(Q: inout [StateAction: Double],
                              returns: inout [StateAction: [Int]],
                              policy: inout [State: Bool]) {
    let blackJack = BlackJack()

    //randomly initialize game state and action
    //blackJack.startGame()
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
    //print(blackJack.stateActionSequence)
    for (index, stateAction) in blackJack.stateActionSequence.reversed().enumerated() {
        let originalIndex = blackJack.stateActionSequence.count - index - 1
        let precedingStateActions = blackJack.stateActionSequence[0..<originalIndex]

        //first-visit
        if !precedingStateActions.contains(stateAction) {
            if returns.keys.contains(stateAction) {
                returns[stateAction]!.append(G)
            }
            else {
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

    //derive state-value function from action-value function
    var v = [State: Double]()
    for (stateAction, value) in Q {
        if v.keys.contains(stateAction.state) {
            v[stateAction.state] = max(v[stateAction.state]!, value)
        }
        else {
            v[stateAction.state] = value
        }
    }
    return (v, policy)
}

func fill_figure_5_2_value_plot(figAndAx: (PythonObject, PythonObject),
                                subplotIndex: (Int, Int),
                                values: [(Int, Int, Double)]){
    let fig = figAndAx.0
    let ax = figAndAx.1

    let valuesDealer = values.map{ $0.0 }
    let valuesSum = values.map{ $0.1 }
    let valuesValue = values.map{ $0.2 }
    let dealerRange = np.arange(1,12, 1)
    let sumRange = np.arange(12,23,1)

    let row = subplotIndex.0
    let col = subplotIndex.1
    let h = ax[row][col].hist2d(valuesDealer, valuesSum, weights: valuesValue,
                                cmap: (col == 0) ? "Greys" : "YlGn",
                                bins: [dealerRange, sumRange],
                                vmin: (col == 0) ? 0 : -1,
                                vmax: 1)

    if row == 1 && col == 0 {
        ax[row][col].set_xlabel("Dealer showing")
        ax[row][col].set_ylabel("(No Useable Ace)\nPlayer sum")
    }
    else if row == 0 && col == 0 {
        ax[row][col].set_title("pi_*")
        ax[row][col].set_ylabel("(Useable Ace)\nPlayer sum")

        ax[row][col].text(0.05, 0.95, "STICK = white\nHIT = black",
                          transform: ax[row][col].transAxes, fontsize: 12,
                          verticalalignment: "top")
    }
    else if row == 0 && col == 1 {
        ax[row][col].set_title("v_*")
    }

    if col == 1 {
        fig.colorbar(h[3], ax: ax[row][col])
    }
}

func make_figure_5_2() {
    print("Generating Figure 5.2")
    let vAndPolicy = run_mc_es(n: 1000000)
    let v = vAndPolicy.0
    let pi = vAndPolicy.1

    let figAndAx = plt.subplots(nrows: 2, ncols: 2, figsize: [8, 6.4]).tuple2

    let vUseableAce = v.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let vNoUseableAce = v.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value) }

    let piUseableAce = pi.filter{ $0.key.useableAce == true && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value == true ? 1.0 : 0.0) }
    let piNoUseableAce = pi.filter{ $0.key.useableAce == false && $0.key.sum <= 21 }.map{ ($0.key.dealer, $0.key.sum, $0.value == true ? 1.0 : 0.0) }

    fill_figure_5_2_value_plot(figAndAx: figAndAx, subplotIndex: (0, 1), values: vUseableAce)
    fill_figure_5_2_value_plot(figAndAx: figAndAx, subplotIndex: (1, 1), values: vNoUseableAce)
    fill_figure_5_2_value_plot(figAndAx: figAndAx, subplotIndex: (0, 0), values: piUseableAce)
    fill_figure_5_2_value_plot(figAndAx: figAndAx, subplotIndex: (1, 0), values: piNoUseableAce)

    plt.savefig("Fig_5.2.png")
}

make_figure_5_1()
make_figure_5_2()
