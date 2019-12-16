class GamblersProblem {

    let n: Int
    let pHead: Double

    init(n: Int, pHead: Double) {
        self.n = n
        self.pHead = pHead
    }

    func nextValue(state: Int, v: [Double]) -> (Double, [Int]) {
        // state is the current capital of the gambler
        let i = state - 1

        // possible wagers
        let actions = Array(0...min(state, self.n+1-state))

        var bestValue = 0.0
        var bestActions = [Int]()
        for a in actions {
            let headValue = (state + a == 100) ? self.pHead * (1 + 0) : self.pHead * v[i + a]
            let tailValue = (state - a == 0) ? 0 : (1-self.pHead) * v[i - a]
            let expectedValue = headValue + tailValue
            if expectedValue > bestValue {
                bestValue = expectedValue
            }
            // give some slack to the best value
            if abs(expectedValue - bestValue) < 1e-5 {
                bestActions.append(a)
            }
        }
        return (bestValue, bestActions)
    }
}
