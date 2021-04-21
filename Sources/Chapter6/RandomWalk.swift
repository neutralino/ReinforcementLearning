enum State {
    case A
    case B
    case C
    case D
    case E
    case terminal
}

struct StateReward {
    let state: State
    let reward: Double

    init(_ s: State, _ r: Double) {
        state = s
        reward = r
    }
}

class RandomWalk {
    let MRP = [ State.A: [State.terminal, State.B],
                State.B: [State.A, State.C],
                State.C: [State.B, State.D],
                State.D: [State.C, State.E],
                State.E: [State.D, State.terminal] ]

    var currentState: State
    var stateRewardSequence = [StateReward]()

    init() {
        currentState = State.C
    }

    func getNextState(s: State) -> State {
        return MRP[s]!.randomElement()!
    }

    func getReward(s1: State, s2: State) -> Double {
        if s1 == State.E && s2 == State.terminal {
            return 1.0
        }
        return 0.0
    }

    func generateStateRewardSequence() {
        while currentState != State.terminal {
            let s1 = currentState
            let s2 = getNextState(s: s1)
            let reward = getReward(s1: s1, s2: s2)
            stateRewardSequence.append(StateReward(currentState, reward))
            currentState = s2
        }
    }
}
