class CarRental {
    let RequestDistLoc1: PoissonDistribution
    let RequestDistLoc2: PoissonDistribution
    let ReturnDistLoc1: PoissonDistribution
    let ReturnDistLoc2: PoissonDistribution
    let maxCars: Int
    let maxTransfer = 5

    init(maxCars: Int, request1Mean: Int, request2Mean: Int,
         return1Mean: Int, return2Mean: Int,
         initialState: (Int, Int)) {
        self.maxCars = maxCars
        self.RequestDistLoc1 = PoissonDistribution(mean: Float(request1Mean))
        self.RequestDistLoc2 = PoissonDistribution(mean: Float(request2Mean))
        self.ReturnDistLoc1 = PoissonDistribution(mean: Float(return1Mean))
        self.ReturnDistLoc2 = PoissonDistribution(mean: Float(return2Mean))
    }

    func computeNextStateAndReward(state1: Int,
                                   state2: Int,
                                   action: Int,
                                   request1: Int,
                                   request2: Int,
                                   return1: Int,
                                   return2: Int) -> (Int, Int, Int) {

        // first we perform the car transfer
        let cappedAction = abs(action) > maxTransfer ? maxTransfer * action.signum() : action
        let carsTransferred = abs(cappedAction)
        let intermediateState1 = state1 - cappedAction
        let intermediateState2 = state2 + cappedAction

        // then we perform the car request
        let cappedRequest1 = min(request1, intermediateState1)
        let cappedRequest2 = min(request2, intermediateState2)
        let carsRented = cappedRequest1 + cappedRequest2

        // then perform the car return
        let newState1 = min(intermediateState1 - cappedRequest1 + return1, self.maxCars)
        let newState2 = min(intermediateState2 - cappedRequest2 + return2, self.maxCars)

        // finally compute the reward
        let reward = carsRented * 10 - carsTransferred * 2
        return (newState1, newState2, reward)
    }
}
