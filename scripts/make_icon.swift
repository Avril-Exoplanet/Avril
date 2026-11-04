import AppKit
import Foundation

guard CommandLine.arguments.count == 2 else {
    fatalError("usage: make_icon.swift BUILD_ROOT")
}

let buildRoot = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let iconset = buildRoot.appendingPathComponent("AvrilNova.iconset", isDirectory: true)
try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

func drawIcon(size: Int) throws -> Data {
    let side = CGFloat(size)
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: size,
        pixelsHigh: size,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else { throw CocoaError(.fileWriteUnknown) }

    bitmap.size = NSSize(width: side, height: side)
    NSGraphicsContext.saveGraphicsState()
    let context = NSGraphicsContext(bitmapImageRep: bitmap)!
    NSGraphicsContext.current = context

    let outer = NSBezierPath(roundedRect: NSRect(x: side * 0.035, y: side * 0.035, width: side * 0.93, height: side * 0.93), xRadius: side * 0.22, yRadius: side * 0.22)
    NSGradient(colors: [
        NSColor(calibratedRed: 0.025, green: 0.085, blue: 0.17, alpha: 1),
        NSColor(calibratedRed: 0.035, green: 0.24, blue: 0.46, alpha: 1),
    ])!.draw(in: outer, angle: -35)

    NSGraphicsContext.saveGraphicsState()
    outer.addClip()
    NSColor.white.withAlphaComponent(0.085).setStroke()
    let grid = NSBezierPath()
    grid.lineWidth = max(0.5, side * 0.0012)
    let step = side / 9
    for index in 1..<9 {
        let position = CGFloat(index) * step
        grid.move(to: NSPoint(x: position, y: side * 0.04))
        grid.line(to: NSPoint(x: position, y: side * 0.96))
        grid.move(to: NSPoint(x: side * 0.04, y: position))
        grid.line(to: NSPoint(x: side * 0.96, y: position))
    }
    grid.stroke()
    NSGraphicsContext.restoreGraphicsState()

    let planetRect = NSRect(x: side * 0.29, y: side * 0.27, width: side * 0.42, height: side * 0.42)
    let planet = NSBezierPath(ovalIn: planetRect)
    NSGradient(colors: [
        NSColor(calibratedRed: 0.12, green: 0.74, blue: 0.88, alpha: 1),
        NSColor(calibratedRed: 0.08, green: 0.28, blue: 0.72, alpha: 1),
    ])!.draw(in: planet, relativeCenterPosition: NSPoint(x: -0.35, y: 0.4))

    NSColor(calibratedRed: 0.62, green: 0.95, blue: 1, alpha: 0.56).setStroke()
    let latitude = NSBezierPath(ovalIn: NSRect(x: side * 0.34, y: side * 0.39, width: side * 0.32, height: side * 0.12))
    latitude.lineWidth = max(1, side * 0.009)
    latitude.stroke()

    NSGraphicsContext.saveGraphicsState()
    let transform = NSAffineTransform()
    transform.translateX(by: side * 0.5, yBy: side * 0.5)
    transform.rotate(byDegrees: -24)
    transform.concat()
    let orbit = NSBezierPath(ovalIn: NSRect(x: -side * 0.39, y: -side * 0.19, width: side * 0.78, height: side * 0.38))
    orbit.lineWidth = max(1.5, side * 0.018)
    orbit.lineCapStyle = .round
    NSColor(calibratedRed: 0.77, green: 0.96, blue: 1, alpha: 0.92).setStroke()
    orbit.stroke()
    NSGraphicsContext.restoreGraphicsState()

    let satellite = NSBezierPath(ovalIn: NSRect(x: side * 0.695, y: side * 0.645, width: side * 0.095, height: side * 0.095))
    NSColor(calibratedRed: 0.48, green: 0.95, blue: 0.93, alpha: 1).setFill()
    satellite.fill()
    NSColor.white.withAlphaComponent(0.9).setStroke()
    satellite.lineWidth = max(1, side * 0.008)
    satellite.stroke()

    NSColor.white.withAlphaComponent(0.32).setStroke()
    outer.lineWidth = max(1, side * 0.006)
    outer.stroke()

    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        throw CocoaError(.fileWriteUnknown)
    }
    return png
}

let files: [(String, Int)] = [
    ("icon_16x16.png", 16), ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32), ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128), ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256), ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512), ("icon_512x512@2x.png", 1024),
]

for (name, size) in files {
    try drawIcon(size: size).write(to: iconset.appendingPathComponent(name), options: .atomic)
}

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", buildRoot.appendingPathComponent("AvrilNova.icns").path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else { fatalError("iconutil failed") }
