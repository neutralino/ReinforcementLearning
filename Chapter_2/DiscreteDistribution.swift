import Foundation
import GameplayKit

class DiscreteDistribution {
    private let randomSource: GKRandomSource
    var distribution: [Double]
    var unitPartition: [Double] = []

    init(randomSource: GKRandomSource, distribution: [Double]) {
        self.randomSource = randomSource
        self.distribution = distribution
        var lowerBound = 0.0
        for i in 0...distribution.count-1 {
            lowerBound += distribution[i]
            self.unitPartition.append(min(lowerBound, 1.0))
        }
    }

    func nextInt() -> Int {
        let x = randomSource.nextUniform() // a random number between 0 and 1
        var index = self.unitPartition.count-1
        for i in 0...self.unitPartition.count-1 {
            if Double(x) < self.unitPartition[i] {
                index = i
                break
            }
        }
        return index
    }
}
