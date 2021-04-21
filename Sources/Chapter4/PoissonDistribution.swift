import GameplayKit

// Generates Poisson distribution via:
// http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf
struct PoissonDistribution {
    private let randomSource: GKRandomSource
    let lambda: Float

    init(mean: Float) {
        precondition(mean > 0)
        self.randomSource = GKRandomSource()
        self.lambda = mean
    }

    func nextInt() -> Int {
        var X = 0
        var P: Float = 1.0
        var u = randomSource.nextUniform() // a random number between 0 and 1
        var finished = false
        while !finished {
            P = u * P
            if P < exp(-lambda) {
                finished = true
            } else {
                u = randomSource.nextUniform()
                X += 1
            }
        }
        return X
    }

    func factorial(n: Int) -> Int {
        return n == 0 ? 1 : n * factorial(n: n - 1)
    }

    func prob(n: Int) -> Double {
        return Double(pow(self.lambda, Float(n)) * exp(-self.lambda) / Float(factorial(n: n)))
    }
}
