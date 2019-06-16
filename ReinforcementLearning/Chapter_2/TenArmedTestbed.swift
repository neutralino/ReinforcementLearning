//
//  TenArmedTestbed.swift
//  ReinforcementLearning
//
//  Created by Don Teo on 2019-05-18.
//  Copyright © 2019 Don Teo. All rights reserved.
//

import Foundation
import GameplayKit

let random = GKRandomSource()

struct TenArmedTestbed {
    let mean = 0
    var arms = [GaussianDistribution]()
    let expectedRewardDistribution: GaussianDistribution
    var indicesOfOptimalAction = [Int]()
    init(mean: Float) {
        // initialize arms
        self.expectedRewardDistribution = GaussianDistribution(randomSource: random, mean: mean, deviation: 1)
        for _ in 0...9 {
            let expectedReward = expectedRewardDistribution.nextFloat()
            self.arms.append(GaussianDistribution(randomSource: random, mean: expectedReward, deviation: 1))
        }

        // find optimal action
        let optimalAction = self.arms.max(by: { (a, b) -> Bool in a.mean < b.mean })
        indicesOfOptimalAction = self.arms.indices.filter { self.arms[$0].mean == optimalAction?.mean }
    }

    func printStats() {
        for i in 0...9 {
            print("Arm \(i): Mean = \(self.arms[i].mean)\t Dev = \(self.arms[i].deviation)")
        }
        print("Optimal action: ")
        //print("\t mean = \(String(describing: optimalAction?.mean))")
        print("\t index = \(indicesOfOptimalAction)")
    }

}
