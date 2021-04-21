enum IVAction {
    case left
    case right
}
struct IVStateAction {
    let state: Bool     // if true, then terminal state is reached
    let action: IVAction

    init(_ s: Bool, _ a: IVAction) {
        state = s
        action = a
    }
}

class InfiniteVariance {
    let behaviorPolicy = [IVAction.left: 0.5, IVAction.right: 0.5]
    var stateActionSequence = [IVStateAction]()

    func nextStateAndReward(action: IVAction) -> (Bool, Int) {
        if action == IVAction.left {
            if Double.random(in: 0...1) < 0.1 {
                return (true, 1)
            } else {
                return (false, 0)
            }
        } else {
            return (true, 0)
        }
    }

    func getNextStateAndReward(policy: [IVAction: Double]) -> (Bool, Int) {
        if Double.random(in: 0...1) <= policy[IVAction.left]! {
            stateActionSequence.append(IVStateAction(false, IVAction.left))
            return nextStateAndReward(action: IVAction.left)
        }
        stateActionSequence.append(IVStateAction(false, IVAction.right))
        return nextStateAndReward(action: IVAction.right)
    }

    func getNextStateAndRewardBehaviorPolicy() -> (Bool, Int) {
        return getNextStateAndReward(policy: behaviorPolicy)
    }
}
