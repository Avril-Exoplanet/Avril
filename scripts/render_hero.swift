#!/usr/bin/env swift

import AppKit
import Foundation

guard CommandLine.arguments.count == 3 else {
    fputs("usage: render_hero.swift <scene.png> <output.png>\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
guard let source = NSImage(contentsOf: inputURL) else {
    fputs("unable to read input image\n", stderr)
    exit(1)
}

let size = NSSize(width: 1_600, height: 720)
let canvas = NSImage(size: size)
canvas.lockFocus()

NSColor(calibratedRed: 0.015, green: 0.035, blue: 0.055, alpha: 1).setFill()
NSRect(origin: .zero, size: size).fill()

let sourceRatio = source.size.width / source.size.height
let targetRatio = size.width / size.height
var crop = NSRect(origin: .zero, size: source.size)
if sourceRatio < targetRatio {
    let cropHeight = source.size.width / targetRatio
    crop.origin.y = (source.size.height - cropHeight) * 0.5
    crop.size.height = cropHeight
} else {
    let cropWidth = source.size.height * targetRatio
    crop.origin.x = (source.size.width - cropWidth) * 0.5
    crop.size.width = cropWidth
}
source.draw(
    in: NSRect(origin: .zero, size: size),
    from: crop,
    operation: .sourceOver,
    fraction: 1,
    respectFlipped: false,
    hints: [.interpolation: NSImageInterpolation.high]
)

let gradient = NSGradient(colorsAndLocations:
    (NSColor(calibratedRed: 0.015, green: 0.035, blue: 0.055, alpha: 1.00), 0.00),
    (NSColor(calibratedRed: 0.015, green: 0.035, blue: 0.055, alpha: 0.98), 0.27),
    (NSColor(calibratedRed: 0.015, green: 0.035, blue: 0.055, alpha: 0.58), 0.48),
    (NSColor(calibratedRed: 0.015, green: 0.035, blue: 0.055, alpha: 0.00), 0.73)
)
gradient?.draw(in: NSRect(x: 0, y: 0, width: 1_190, height: 720), angle: 0)

NSColor(calibratedRed: 0.76, green: 0.11, blue: 0.16, alpha: 1).setFill()
NSRect(x: 0, y: 708, width: 1_600, height: 12).fill()

func font(_ name: String, size: CGFloat, fallbackWeight: NSFont.Weight) -> NSFont {
    NSFont(name: name, size: size) ?? NSFont.systemFont(ofSize: size, weight: fallbackWeight)
}

func draw(_ text: String, at point: NSPoint, attributes: [NSAttributedString.Key: Any]) {
    NSAttributedString(string: text, attributes: attributes).draw(at: point)
}

let white = NSColor(calibratedWhite: 0.97, alpha: 1)
let muted = NSColor(calibratedRed: 0.69, green: 0.78, blue: 0.84, alpha: 1)
let cyan = NSColor(calibratedRed: 0.35, green: 0.78, blue: 0.92, alpha: 1)
let red = NSColor(calibratedRed: 0.92, green: 0.28, blue: 0.32, alpha: 1)

draw(
    "AVRIL   /   EXOPLANET WORKSTATION",
    at: NSPoint(x: 72, y: 630),
    attributes: [
        .font: font("SFMono-Semibold", size: 15, fallbackWeight: .semibold),
        .foregroundColor: cyan,
        .kern: 2.2,
    ]
)

draw(
    "Find the signal.\nKeep the evidence.",
    at: NSPoint(x: 68, y: 388),
    attributes: [
        .font: font("HelveticaNeue-Medium", size: 66, fallbackWeight: .medium),
        .foregroundColor: white,
        .kern: -1.4,
        .paragraphStyle: {
            let style = NSMutableParagraphStyle()
            style.lineSpacing = -4
            return style
        }(),
    ]
)

draw(
    "Evidence-preserving transit-candidate search for public TESS data.",
    at: NSPoint(x: 72, y: 332),
    attributes: [
        .font: font("HelveticaNeue", size: 22, fallbackWeight: .regular),
        .foregroundColor: muted,
    ]
)

let steps = ["TARGET PRIOR", "TESS DATA", "AVRIL V3", "SCIENCE REVIEW"]
var x: CGFloat = 72
for (index, step) in steps.enumerated() {
    let accent = index == 2 ? red : cyan
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font("SFMono-Medium", size: 13, fallbackWeight: .medium),
        .foregroundColor: accent,
        .kern: 1.1,
    ]
    draw(step, at: NSPoint(x: x, y: 252), attributes: attrs)
    x += CGFloat(step.count) * 9.0 + 34
    if index < steps.count - 1 {
        draw("→", at: NSPoint(x: x - 23, y: 250), attributes: [
            .font: font("HelveticaNeue", size: 18, fallbackWeight: .regular),
            .foregroundColor: NSColor(calibratedWhite: 0.72, alpha: 0.8),
        ])
    }
}

draw(
    "LOCAL-FIRST  ·  AUDITABLE  ·  CANDIDATES REMAIN CANDIDATES",
    at: NSPoint(x: 72, y: 68),
    attributes: [
        .font: font("SFMono-Medium", size: 12, fallbackWeight: .medium),
        .foregroundColor: NSColor(calibratedWhite: 0.74, alpha: 0.95),
        .kern: 1.4,
    ]
)

canvas.unlockFocus()

guard let tiff = canvas.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [.compressionFactor: 0.92])
else {
    fputs("unable to render PNG\n", stderr)
    exit(1)
}

try png.write(to: outputURL, options: .atomic)
