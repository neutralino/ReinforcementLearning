enum Suit : CaseIterable {
    case spade
    case heart
    case club
    case diamond
}

struct Card {
    let value: Int
    let suit: Suit

    init(_ value: Int, _ suit: Suit) {
        self.value  = value
        self.suit = suit
    }
}

// the state is the player's current sum (12-21), the dealer's one
// showing (1-10) card and whether or not player holds a usable
// ace.  Note that if the player's current sum is less than or
// equal to 11, then there is no decision to be made, since
// hitting will have no negative consequences.  So the number of
// states is really 10x10x2 = 200.
struct State: Hashable {
    let sum: Int
    let dealer: Int
    let useableAce: Bool

    init(_ s: Int, _ d: Int, _ u: Bool) {
        sum = s
        dealer = d
        useableAce = u
    }
}

extension State: Equatable {
  static func == (lhs: State, rhs: State) -> Bool {
    return lhs.sum == rhs.sum &&
      lhs.dealer == rhs.dealer &&
      lhs.useableAce == rhs.useableAce
  }
}

struct StateAction: Hashable {
    let state: State
    let action: Bool

    init(_ s: State, _ a: Bool) {
        state = s
        action = a
    }
}

extension StateAction: Equatable {
  static func == (lhs: StateAction, rhs: StateAction) -> Bool {
    return lhs.state == rhs.state &&
      lhs.action == rhs.action
  }
}

class BlackJack {
    var dealerCards = [Card]()
    var playerCards = [Card]()
    var playerNUseableAces: Int = 0
    var sequence = [State]()
    var stateActionSequence = [StateAction]()
    var currentState: State = State(0, 0, false)

    func updateCurrentState() {
        currentState = State(computePlayerHand(),
                             min(viewDealerHand().value, 10),
                             playerHasUsableAce())
    }

    // draw card from deck with replacement
    func drawCard() -> Card {
        return Card(Int.random(in: 1...13), Suit.allCases.randomElement()!)
    }

    func startGame(){
        playerCards.append(drawCard())
        playerCards.append(drawCard())
        dealerCards.append(drawCard())
        dealerCards.append(drawCard())
        updateCurrentState()
        sequence.append(currentState)
    }

    // for Monte Carlo ES, to get better stats on some states
    func startRandomStateGame() {
        let sum = Int.random(in: 12...21)
        let dealer = Int.random(in: 1...10)
        let useableAce = (sum == 21) ? true : Bool.random()

        // pick any random cards that satisfy this state
        dealerCards.append(Card(dealer, Suit.allCases.randomElement()!))
        dealerCards.append(drawCard())
        if useableAce {
            playerCards.append(Card(1, Suit.allCases.randomElement()!))
            playerCards.append(Card(sum-11, Suit.allCases.randomElement()!))
        }
        else {
            playerCards.append(Card(10, Suit.allCases.randomElement()!))
            playerCards.append(Card(sum-10, Suit.allCases.randomElement()!))
        }
        updateCurrentState()
        sequence.append(currentState)
    }

    func initializeStateActionSequence(action: Bool) {
        stateActionSequence.append(StateAction(currentState, action))
        if action {
            playerHit()
        }
    }

    func playerHit(){
        playerCards.append(drawCard())
    }

    func dealerHit(){
        dealerCards.append(drawCard())
    }

    func viewDealerHand() -> Card {
        return dealerCards.first!
    }

    func sumWithNAces(hand: [Card], nAces: Int) -> Int {
        var sum = 0
        var aceCounter = nAces

        for card in hand {
            if card.value == 1 {
                if aceCounter > 0 {
                    sum += 11
                    aceCounter -= 1
                }
                else {
                    sum += 1
                }
            }
            else if card.value >= 10 {
                sum += 10
            }
            else {
                sum += card.value
            }
        }
        return sum
    }

    func computeSum(hand: [Card]) -> Int {
        var sum = 0
        // if there are aces, treat them as 11's. If the sum exceeds
        // 21, back the aces off one by one to 1's.
        let nAce = nAces(hand: hand)
        for i in (0...nAce).reversed() {
            sum = sumWithNAces(hand: hand, nAces: i)
            if sum <= 21 {
                playerNUseableAces = i
                break
            }
        }
        return sum
    }

    func computeDealerHand() -> Int {
        return computeSum(hand: dealerCards)
    }

    func computePlayerHand() -> Int {
        return computeSum(hand: playerCards)
    }

    func nAces(hand: [Card]) -> Int {
        return hand.filter{ $0.value==1 }.count
    }

    func dealerNAces() -> Int {
        return nAces(hand: dealerCards)
    }

    func playerNAces() -> Int {
        return nAces(hand: playerCards)
    }

    func playerHasUsableAce() -> Bool {
        let _ = computePlayerHand()
        return playerNUseableAces > 0
    }


    func runPlayerPolicy(policy: [State: Bool]) {
        updateCurrentState()
        var state = currentState
        while policy.keys.contains(state) && policy[state]! {
            playerHit()
            stateActionSequence.append(StateAction(state, true))
            updateCurrentState()
            state = currentState
            sequence.append(state)
        }
        stateActionSequence.append(StateAction(state, false))
    }

    // dealer policy is to hit if the sum is less than 17
    func runDealerPolicy() {
        var sum = computeDealerHand()
        while sum <= 16 {
            dealerHit()
            sum = computeDealerHand()
        }
    }

    func computeReward() -> Int {
        let playerSum = computePlayerHand()
        let dealerSum = computeDealerHand()

        if playerSum > 21 {
            return -1
        }
        else if dealerSum > 21 {
            return +1
        }
        else if playerSum > dealerSum {
            return +1
        }
        else if playerSum == dealerSum {
            return 0
        }
        return -1
    }
}
