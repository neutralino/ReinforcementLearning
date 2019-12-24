// NB: Point(0, 0) is the top-left corner in this chapter.
struct Point: Equatable {
    var x: Int
    var y: Int
    init(_ x: Int, _ y: Int) {
        self.x = x
        self.y = y
    }
}

enum Action {
    case north
    case south
    case east
    case west
}

struct GridWorld {
    var n: Int

    init(_ n: Int) {
        self.n = n
    }

    func intToGrid(_ i: Int) -> Point {
        precondition(i < n * n, "Index must map to valid grid state.")
        return Point(i % n, i / n)
    }

    func gridToInt(_ p: Point) -> Int {
        return n * p.y + p.x
    }

    // return subsequent state and reward
    func move(_ p: Point, _ action: Action) -> (Point, Int) {
        if p == Point(0, 0) {
            return (Point(0, 0), 0)
        }
        else if p == Point(n-1, n-1) {
            return (Point(0, 0), 0)
        }
        let reward = -1
        switch action {
        case .north:
            return (Point(p.x, min(p.y+1, n-1)), reward)
        case .south:
            return (Point(p.x, max(p.y-1, 0)), reward)
        case .east:
            return (Point(min(p.x+1, n-1), p.y), reward)
        case .west:
            return (Point(max(p.x-1, 0), p.y), reward)
        }
    }

    func getNextStatesAndRewards(_ p: Point) -> (Point, Point, Point, Point, Int, Int, Int, Int) {
        let (northState, northReward) = move(p, Action.north)
        let (southState, southReward) = move(p, Action.south)
        let (eastState, eastReward) = move(p, Action.east)
        let (westState, westReward) = move(p, Action.west)
        return (northState, southState, eastState, westState, northReward, southReward, eastReward, westReward)
    }
}
