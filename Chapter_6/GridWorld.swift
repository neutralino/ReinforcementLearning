// NB: Point(0, 0) is the bottom-left corner in this chapter.
struct Point: Equatable {
    var x: Int
    var y: Int

    public var description: String { return "Point(\(x), \(y))" }

    init(_ x: Int, _ y: Int) {
        self.x = x
        self.y = y
    }

}

enum Action: Int {
    case north
    case south
    case east
    case west
}

struct GridWorld {
    var nX: Int
    var nY: Int

    // Wind strength
    let windStrength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    init(_ nX: Int, _ nY: Int) {
        self.nX = nX
        self.nY = nY
    }

    func intToGrid(_ i: Int) -> Point {
        precondition(i < nX * nY, "Index must map to valid grid state.")
        return Point(i % nX, i / nX)
    }

    func gridToInt(_ p: Point) -> Int {
        return nX * p.y + p.x
    }

    func addWind(_ p: Point, _ nextP: Point) -> Point {
        return Point(nextP.x, nextP.y + windStrength[p.x])
    }

    func keepOnGrid(_ p: Point) -> Point {
        let x = max(min(p.x, nX-1), 0)
        let y = max(min(p.y, nY-1), 0)
        return Point(x, y)
    }

    // return sebsequent state and reward
    func move(_ p: Point, _ action: Action) -> (Point, Int) {
        var nextP: Point
        switch action {
        case .north:
            nextP = Point(p.x, p.y+1)
        case .south:
            nextP = Point(p.x, p.y-1)
        case .east:
            nextP = Point(p.x+1, p.y)
        case .west:
            nextP = Point(p.x-1, p.y)
        }
        nextP = keepOnGrid(nextP)
        nextP = addWind(p, nextP)
        nextP = keepOnGrid(nextP)
        if nextP == Point(7, 3) {
            return (nextP, 0)
        }
        return (nextP, -1)
    }
}
