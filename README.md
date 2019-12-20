# Reinforcement Learning: An Introduction

(work in progress)

Swift replication (with some Python interoperability) for Sutton & Barto's book: [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/the-book-2nd.html).

Last tested with Swift 5.0 using the Swift for Tensorflow toolchain (`swift-tensorflow-RELEASE-0.3.1.xctoolchain`) and Python 3.7.

On macOS, to run the code for a given chapter, compile all files in the chapter folder, e.g.
```
$ cd ReinforcementLearning/Chapter_2
$ swiftc -O -sdk `xcrun --show-sdk-path` main.swift TenArmedTestbed.swift GaussianDistribution.swift DiscreteDistribution.swift -o program
$ ./program
```

# Contents

| Chapter | Figures |
| ------- | ------- |
| 2: Multi-armed Bandits | [2.2](ReinforcementLearning/Chapter_2/Fig_2.2.png), [2.3](ReinforcementLearning/Chapter_2/Fig_2.3.png), [2.4](ReinforcementLearning/Chapter_2/Fig_2.4.png), [2.5](ReinforcementLearning/Chapter_2/Fig_2.5.png), [2.6](ReinforcementLearning/Chapter_2/Fig_2.6.png) |
| 3: Finite Markov Decision Processes | [3.2](ReinforcementLearning/Chapter_3/Fig_3.2.png), [3.5](ReinforcementLearning/Chapter_3/Fig_3.5.png) |
| 4: Dynamic Programming | [4.1](ReinforcementLearning/Chapter_4/Fig_4.1.png), [4.2](ReinforcementLearning/Chapter_4/Fig_4.2.png), [4.3](ReinforcementLearning/Chapter_4/Fig_4.3.png) |
| 5: Monte Carlo Methods | [5.1](ReinforcementLearning/Chapter_5/Fig_5.1.png), [5.2](ReinforcementLearning/Chapter_5/Fig_5.2.png), [5.3](ReinforcementLearning/Chapter_5/Fig_5.3.png), [5.4](ReinforcementLearning/Chapter_5/Fig_5.4.png) |
