// NB: Point(0, 0) is the bottom-left corner in this chapter.
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

    // return sebsequent state and reward
    func move(_ p: Point, _ action: Action) -> (Point, Int) {
        if p == Point(1, 4) {
            return (Point(1, 0), 10)
        }
        else if p == Point(3, 4){
            return (Point(3, 2), 5)
        }
        switch action {
        case .north:
            if p.y+1 > n-1 {
                return (Point(p.x, p.y), -1)
            }
            else {
                return (Point(p.x, p.y+1), 0)
            }
        case .south:
            if p.y-1 < 0 {
                return (Point(p.x, p.y), -1)
            }
            else {
                return (Point(p.x, p.y-1), 0)
            }
        case .east:
            if p.x+1 > n-1 {
                return (Point(p.x, p.y), -1)
            }
            else {
                return (Point(p.x+1, p.y), 0)
            }
        case .west:
            if p.x-1 < 0 {
                return (Point(p.x, p.y), -1)
            }
            else {
                return (Point(p.x-1, p.y), 0)
            }
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
