// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "ReinforcementLearning",
  platforms: [
    .macOS(.v10_11)
  ],
  dependencies: [
    // Dependencies declare other packages that this package depends on.
    // .package(url: /* package url */, from: "1.0.0"),
    .package(name: "SwiftPlot", url: "https://github.com/KarthikRIyer/swiftplot.git", from: "2.0.0"),
    .package(name: "LANumerics", url: "https://github.com/phlegmaticprogrammer/LANumerics.git", from: "0.1.11")
  ],
  targets: [
    // Targets are the basic building blocks of a package. A target can define a module or a test suite.
    // Targets can depend on other targets in this package, and on products in packages this package depends on.
    .target(
      name: "ReinforcementLearning",
      dependencies: ["LANumerics", "SwiftPlot",
                     .product(name:"AGGRenderer", package: "SwiftPlot")]),
    .target(
      name: "Chapter2",
      dependencies: [
        "SwiftPlot",
       .product(name:"AGGRenderer", package: "SwiftPlot")],
      exclude: ["Output"]),
    .target(
      name: "Chapter3",
      dependencies: [
        "LANumerics", "SwiftPlot",
       .product(name:"AGGRenderer", package: "SwiftPlot")],
      exclude: ["Output"]),
    .target(
      name: "Chapter4",
      dependencies: [
        "LANumerics", "SwiftPlot",
       .product(name:"AGGRenderer", package: "SwiftPlot")],
      exclude: ["Output"]),
    .target(
      name: "Chapter5",
      dependencies: [
        "SwiftPlot",
       .product(name:"AGGRenderer", package: "SwiftPlot")],
      exclude: ["Output"]),
    .target(
      name: "Chapter6",
      dependencies: [
        "SwiftPlot",
       .product(name:"AGGRenderer", package: "SwiftPlot")],
      exclude: ["Output"]),
    .testTarget(
      name: "ReinforcementLearningTests",
      dependencies: ["ReinforcementLearning"]),
  ]
)
