import Foundation
import Python

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")

let stateToIndex = [State.A: 0, State.B: 1, State.C: 2, State.D: 3, State.E: 4]
let trueV = Array(1...5).map{ Double($0) / 6.0 }

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
func run_mc_episode(alpha: Double, V: inout [Double]){
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

func fill_value_estimate_subplot(ax: PythonObject, initialV: [Double]) {
    let l = Array(1...5)
    var V = initialV
    ax[0].plot(l, trueV, marker: "o", markersize: 3, label: "True value")
    ax[0].plot(l, V, marker: "o", markersize: 3, label: "0 Episodes")

    let alpha = 0.1

    for nEpisodes in [1, 10, 100] {
        for _ in 0..<nEpisodes {
            run_td0_episode(alpha: alpha, V: &V)
        }
        ax[0].plot(l, V, marker: "o", markersize: 3, label: "\(nEpisodes) Episodes")
    }

    ax[0].set_title("Estimated value")
    ax[0].set_xticks(l)
    ax[0].set_xticklabels(["A", "B", "C", "D", "E"])
    ax[0].set_xlabel("State")
    ax[0].set_ylim([0,1])
    ax[0].legend()
}

func generateValueEstimates(initialV: [Double], alpha: Double, nRuns: Int, nEpisodes: Int, TD: Bool) -> [[[Double]]] {
    var allVsRuns = [[[Double]]]()

    for _ in 0..<nRuns {
        var V = initialV
        var allVs = [[Double]]()
        for _ in 0..<nEpisodes {
            if TD {
                run_td0_episode(alpha: alpha, V: &V)
            }
            else {
                run_mc_episode(alpha: alpha, V: &V)
            }
            allVs.append(V)
        }
        allVsRuns.append(allVs)
    }
    return allVsRuns
}

func computeRMSE(allVsRuns: [[[Double]]]) -> [Double]{
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

    let RMSEs = MSEs.map{ sqrt($0) }
    return RMSEs
}

func fill_learning_curve_subplot(ax: PythonObject, initialV: [Double]) {
    let nRuns = 100
    let nEpisodes = 100

    for alpha in [0.05, 0.1, 0.15] {
        let allVsRuns = generateValueEstimates(initialV: initialV, alpha: alpha,
                                               nRuns: nRuns, nEpisodes: nEpisodes,
                                               TD: true)
        let RMSEs = computeRMSE(allVsRuns: allVsRuns)

        ax[1].plot(Array(0..<nEpisodes), RMSEs, label: "TD: alpha=\(alpha)")
    }

    for alpha in [0.01, 0.02, 0.03, 0.04] {
        let allVsRuns = generateValueEstimates(initialV: initialV, alpha: alpha,
                                               nRuns: nRuns, nEpisodes: nEpisodes,
                                               TD: false)
        let RMSEs = computeRMSE(allVsRuns: allVsRuns)

        ax[1].plot(Array(0..<nEpisodes), RMSEs, label: "MC: alpha=\(alpha)")
    }

    ax[1].set_title("RMS error, averaged over states")
    ax[1].set_xlabel("Episodes")
    ax[1].set_ylim([0, 0.25])
    ax[1].legend()
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
            }
            else {
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

    let figAndAx = plt.subplots(1, 2, figsize: [10, 5]).tuple2
    let ax = figAndAx.1

    fill_value_estimate_subplot(ax: ax, initialV: V)
    fill_learning_curve_subplot(ax: ax, initialV: V)
    plt.savefig("Example_6.2.png")
}

func make_figure_6_2() {
    print("Generating Figure 6.2")
    let _ = plt.figure(figsize: [6.4, 4.8])

    let nRuns = 100
    let nEpisodes = 100
    let alpha = 0.001

    // initial value function
    let V = Array(repeating: 0.5, count: 5)

    var allVsRuns = generateBatchedUpdateEstimates(initialV: V, alpha: alpha, nRuns: nRuns, nEpisodes: nEpisodes, TD: true)
    var RMSEs = computeRMSE(allVsRuns: allVsRuns)
    plt.plot(Array(0..<nEpisodes), RMSEs, label: "TD: alpha=\(alpha)")

    allVsRuns = generateBatchedUpdateEstimates(initialV: V, alpha: alpha, nRuns: nRuns, nEpisodes: nEpisodes, TD: false)
    RMSEs = computeRMSE(allVsRuns: allVsRuns)
    plt.plot(Array(0..<nEpisodes), RMSEs, label: "MC: alpha=\(alpha)")

    plt.title("Batch Training")
    plt.xlabel("Episodes")
    plt.ylabel("RMS error, averaged over states")
    plt.ylim([0.0, 0.25])
    plt.legend()
    plt.savefig("Fig_6.2.png")
}

make_example_6_2()
make_figure_6_2()
