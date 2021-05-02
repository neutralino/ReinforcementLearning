//
//  Generate samples from Gaussian distribution using Box-Muller transform.
//  Copied from https://stackoverflow.com/questions/49470358/how-can-i-return-a-float-or-double-from-normal-distribution-in-swift-4
// (Doesn't look GameplayKit's GKGaussianDistribution can generate floats)

import GameplayKit

class GaussianDistribution {
    private let randomSource: GKRandomSource
    let mean: Float
    let deviation: Float

    init(randomSource: GKRandomSource, mean: Float, deviation: Float) {
        precondition(deviation >= 0)
        self.randomSource = randomSource
        self.mean = mean
        self.deviation = deviation
    }

    func nextFloat() -> Float {
        guard deviation > 0 else { return mean }

        var x1 = randomSource.nextUniform() // a random number between 0 and 1
        let x2 = randomSource.nextUniform() // a random number between 0 and 1

        x1 = x1 > 0 ? x1 : Float.leastNonzeroMagnitude // x1 cannot be exactly 0

        let z1 = sqrt(-2 * log(x1)) * cos(2 * Float.pi * x2) // z1 is normally distributed

        // Convert z1 from the Standard Normal Distribution to our Normal Distribution
        return z1 * deviation + mean
    }
}
