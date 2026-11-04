// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "AvrilNova",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "AvrilNova", targets: ["AvrilNova"]),
    ],
    targets: [
        .binaryTarget(
            name: "CorePlot",
            path: "Vendor/CorePlot.xcframework"
        ),
        .executableTarget(
            name: "AvrilNova",
            dependencies: ["CorePlot"],
            path: "Sources/AvrilNova",
            resources: [
                .copy("../../Resources/tess-allsky-2022-5k.jpg"),
                .copy("../../Resources/webb-carina-cosmic-cliffs-2000.png"),
                .copy("../../Resources/earth-blue-marble.jpg"),
            ],
            linkerSettings: [
                .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"])
            ]
        ),
    ]
)
