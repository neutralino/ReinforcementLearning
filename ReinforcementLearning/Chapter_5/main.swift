import Foundation
import Python

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")

func first_visit_MC_iteration(v: inout [State: Double], returns: inout [State: [Int]]) {
    let blackJack = BlackJack()
    blackJack.startGame()

    blackJack.runPlayerPolicy()
    blackJack.runDealerPolicy()

    // since all rewards within the game are 0, the total return is
    // just the final reward.
    let G = blackJack.computeReward()

    for (index, state) in blackJack.sequence.reversed().enumerated() {
        let original_index = blackJack.sequence.count - index - 1
        let preceding_states = blackJack.sequence[0..<original_index]

        //first-visit
        if !preceding_states.contains(state) {
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
        first_visit_MC_iteration(v: &v, returns: &returns)
    }
    return v
}

func fill_figure_5_1_subplot(ax: PythonObject, subplotIndex: (Int, Int),
                             values: [(Int, Int, Double)]) {
    let dealerRange = np.arange(1,12, 1)
    let sumRange = np.arange(12,23,1)

    let valuesDealer = values.map{ $0.0 }
    let valuesSum = values.map{ $0.1 }
    let valuesValue = values.map{ $0.2 }

    let row = subplotIndex.0
    let col = subplotIndex.1
    let h = ax[row][col].hist2d(valuesDealer, valuesSum, weights: valuesValue, cmap: "YlGn",
                                bins: [dealerRange, sumRange])

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
    if row == 1 && col == 1 {
        plt.colorbar(h[3], ax: ax)
    }
}

func make_figure_5_1() {

    let v10K = run_mc_policy_evaluation(n: 10000)
    let v500K = run_mc_policy_evaluation(n: 500000)

    let figAndAx = plt.subplots(nrows: 2, ncols: 2, figsize: [8, 6.4]).tuple2
    let ax = figAndAx.1
    let v10KUseableAce = v10K.filter{ $0.key.useableAce == true }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let v10KNoUseableAce = v10K.filter{ $0.key.useableAce == true }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let v500KUseableAce = v500K.filter{ $0.key.useableAce == true }.map{ ($0.key.dealer, $0.key.sum, $0.value) }
    let v500KNoUseableAce = v500K.filter{ $0.key.useableAce == true }.map{ ($0.key.dealer, $0.key.sum, $0.value) }

    fill_figure_5_1_subplot(ax: ax, subplotIndex: (0, 0), values: v10KUseableAce)
    fill_figure_5_1_subplot(ax: ax, subplotIndex: (1, 0), values: v10KNoUseableAce)
    fill_figure_5_1_subplot(ax: ax, subplotIndex: (0, 1), values: v500KUseableAce)
    fill_figure_5_1_subplot(ax: ax, subplotIndex: (1, 1), values: v500KNoUseableAce)

    plt.savefig("Fig_5.1.png")
}

make_figure_5_1()
