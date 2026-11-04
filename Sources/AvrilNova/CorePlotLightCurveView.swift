import AppKit
import CorePlot
import SwiftUI

/// Core Plot uses the macOS scroll wheel for panning by default. Scientific
/// time-series tools conventionally reserve drag for pan and wheel for zoom,
/// so this host provides cursor-anchored range scaling while retaining Core
/// Plot's native drag/pinch interaction.
final class ScientificGraphHostingView: CPTGraphHostingView {
    private var homeXRange: CPTPlotRange?
    private var homeYRange: CPTPlotRange?

    func captureHomeRanges() {
        guard let space = hostedGraph?.defaultPlotSpace as? CPTXYPlotSpace else { return }
        homeXRange = space.xRange.copy() as? CPTPlotRange
        homeYRange = space.yRange.copy() as? CPTPlotRange
    }

    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 {
            restoreHomeRanges()
            return
        }
        super.mouseDown(with: event)
    }

    override func scrollWheel(with event: NSEvent) {
        guard let space = hostedGraph?.defaultPlotSpace as? CPTXYPlotSpace,
              let anchor = space.plotPoint(for: event),
              anchor.count >= 2 else {
            super.scrollWheel(with: event)
            return
        }

        let raw = min(max(event.scrollingDeltaY, -22), 22)
        guard raw.isFinite, raw != 0 else { return }
        let gain: CGFloat = event.hasPreciseScrollingDeltas ? 0.012 : 0.075
        let scale = exp(raw * gain)
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        let yOnly = modifiers.contains(.shift)
        let bothAxes = modifiers.contains(.command) || modifiers.contains(.control)

        if !yOnly {
            space.xRange = zoomed(space.xRange, around: anchor[0].doubleValue, scale: scale)
        }
        if yOnly || bothAxes {
            space.yRange = zoomed(space.yRange, around: anchor[1].doubleValue, scale: scale)
        }
    }

    private func zoomed(_ range: CPTPlotRange, around anchor: Double, scale: CGFloat) -> CPTPlotRange {
        let oldLength = range.lengthDouble
        guard oldLength.isFinite, oldLength != 0 else { return range }
        let signedLength = oldLength * Double(scale)
        let minimum = max(abs(oldLength) * 0.000_01, 1e-12)
        let newLength = copysign(max(abs(signedLength), minimum), oldLength)
        let ratio = (anchor - range.locationDouble) / oldLength
        let newLocation = anchor - newLength * ratio
        return CPTPlotRange(
            location: NSNumber(value: newLocation),
            length: NSNumber(value: newLength)
        )
    }

    private func restoreHomeRanges() {
        guard let space = hostedGraph?.defaultPlotSpace as? CPTXYPlotSpace else { return }
        if let homeXRange { space.xRange = homeXRange.copy() as! CPTPlotRange }
        if let homeYRange { space.yRange = homeYRange.copy() as! CPTPlotRange }
    }
}

/// Native Core Plot host for scientific light-curve inspection.
///
/// Core Plot is a mature BSD-licensed Cocoa plotting framework. This wrapper
/// deliberately keeps interaction, axes, rasterization, and scatter rendering
/// inside Core Plot instead of reproducing those systems in application code.
struct CorePlotLightCurveView: NSViewRepresentable {
    let points: [LightCurvePoint]
    let gaps: [LightCurveGap]
    let xAxisTitle: String
    let yAxisTitle: String
    let resetToken: Int

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> ScientificGraphHostingView {
        let view = ScientificGraphHostingView(frame: .zero)
        view.allowPinchScaling = true
        context.coordinator.install(
            in: view,
            points: points,
            gaps: gaps,
            xAxisTitle: xAxisTitle,
            yAxisTitle: yAxisTitle,
            resetToken: resetToken
        )
        return view
    }

    func updateNSView(_ view: ScientificGraphHostingView, context: Context) {
        context.coordinator.install(
            in: view,
            points: points,
            gaps: gaps,
            xAxisTitle: xAxisTitle,
            yAxisTitle: yAxisTitle,
            resetToken: resetToken
        )
    }

    final class Coordinator: NSObject, CPTScatterPlotDataSource {
        private var series: [Int: [LightCurvePoint]] = [:]
        private var signature = 0

        @MainActor
        func install(
            in hostingView: CPTGraphHostingView,
            points: [LightCurvePoint],
            gaps: [LightCurveGap],
            xAxisTitle: String,
            yAxisTitle: String,
            resetToken: Int
        ) {
            let nextSignature = points.count
                ^ (points.first?.time.hashValue ?? 0)
                ^ (points.last?.time.hashValue ?? 0)
                ^ gaps.count
                ^ xAxisTitle.hashValue
                ^ yAxisTitle.hashValue
                ^ resetToken
            guard hostingView.hostedGraph == nil || nextSignature != signature else { return }
            signature = nextSignature
            series = Dictionary(grouping: points.filter { $0.time.isFinite && $0.flux.isFinite }, by: \.sector)

            let graph = CPTXYGraph(frame: .zero)
            graph.fill = CPTFill(color: CPTColor(componentRed: 0.976, green: 0.982, blue: 0.990, alpha: 1))
            graph.paddingLeft = 8
            graph.paddingRight = 12
            graph.paddingTop = 10
            graph.paddingBottom = 8
            graph.plotAreaFrame?.paddingLeft = 58
            graph.plotAreaFrame?.paddingRight = 16
            graph.plotAreaFrame?.paddingTop = 14
            graph.plotAreaFrame?.paddingBottom = 42
            graph.plotAreaFrame?.fill = CPTFill(color: CPTColor(componentRed: 0.992, green: 0.995, blue: 1, alpha: 1))
            graph.plotAreaFrame?.borderLineStyle = borderStyle(width: 0.7, alpha: 0.24)

            guard let plotSpace = graph.defaultPlotSpace as? CPTXYPlotSpace else {
                hostingView.hostedGraph = graph
                (hostingView as? ScientificGraphHostingView)?.captureHomeRanges()
                return
            }
            plotSpace.allowsUserInteraction = true

            let sorted = series.values.flatMap { $0 }
            let xMin = sorted.map(\.time).min() ?? 0
            let xMax = sorted.map(\.time).max() ?? 1
            let fluxes = sorted.map(\.flux).sorted()
            let yLow = percentile(fluxes, fraction: 0.006) ?? 0.99
            let yHigh = percentile(fluxes, fraction: 0.994) ?? 1.01
            let xSpan = max(xMax - xMin, 1e-5)
            let ySpan = max(yHigh - yLow, 1e-5)
            plotSpace.xRange = CPTPlotRange(location: NSNumber(value: xMin - xSpan * 0.012), length: NSNumber(value: xSpan * 1.024))
            plotSpace.yRange = CPTPlotRange(location: NSNumber(value: yLow - ySpan * 0.10), length: NSNumber(value: ySpan * 1.20))

            configureAxes(
                graph: graph,
                xSpan: xSpan,
                ySpan: ySpan,
                xAxisTitle: xAxisTitle,
                yAxisTitle: yAxisTitle
            )

            if let xAxis = (graph.axisSet as? CPTXYAxisSet)?.xAxis {
                let fill = CPTFill(color: CPTColor(componentRed: 0.88, green: 0.23, blue: 0.18, alpha: 0.075))
                for gap in gaps where gap.end > gap.start {
                    xAxis.addBackgroundLimitBand(
                        CPTLimitBand(
                            range: CPTPlotRange(
                                location: NSNumber(value: gap.start),
                                length: NSNumber(value: gap.end - gap.start)
                            ),
                            fill: fill
                        )
                    )
                }
            }

            let palette: [(CGFloat, CGFloat, CGFloat)] = [
                (0.08, 0.43, 0.88), (0.02, 0.64, 0.67), (0.36, 0.29, 0.82),
                (0.90, 0.35, 0.20), (0.11, 0.58, 0.36), (0.79, 0.30, 0.61),
            ]
            for (index, sector) in series.keys.sorted().enumerated() {
                let plot = CPTScatterPlot(frame: .zero)
                plot.identifier = NSNumber(value: sector)
                plot.dataSource = self
                plot.dataLineStyle = nil
                let color = palette[index % palette.count]
                let symbol = CPTPlotSymbol.ellipse()
                symbol.size = CGSize(width: 2.1, height: 2.1)
                symbol.lineStyle = nil
                symbol.fill = CPTFill(color: CPTColor(componentRed: color.0, green: color.1, blue: color.2, alpha: 0.70))
                plot.plotSymbol = symbol
                graph.add(plot)
            }

            hostingView.hostedGraph = graph
            (hostingView as? ScientificGraphHostingView)?.captureHomeRanges()
        }

        func numberOfRecords(for plot: CPTPlot) -> UInt {
            guard let sector = (plot.identifier as? NSNumber)?.intValue else { return 0 }
            return UInt(series[sector]?.count ?? 0)
        }

        func number(for plot: CPTPlot, field: UInt, record: UInt) -> Any? {
            guard let sector = (plot.identifier as? NSNumber)?.intValue,
                  let points = series[sector],
                  Int(record) < points.count,
                  let axis = CPTScatterPlotField(rawValue: Int(field)) else { return nil }
            let point = points[Int(record)]
            switch axis {
            case .X: return NSNumber(value: point.time)
            case .Y: return NSNumber(value: point.flux)
            @unknown default: return nil
            }
        }

        private func configureAxes(
            graph: CPTXYGraph,
            xSpan: Double,
            ySpan: Double,
            xAxisTitle: String,
            yAxisTitle: String
        ) {
            guard let axes = graph.axisSet as? CPTXYAxisSet else { return }
            let axisLine = borderStyle(width: 0.8, alpha: 0.55)
            let gridLine = borderStyle(width: 0.5, alpha: 0.16)
            let label = CPTMutableTextStyle()
            label.color = CPTColor(componentRed: 0.28, green: 0.33, blue: 0.40, alpha: 1)
            label.fontName = "SFMono-Regular"
            label.fontSize = NovaTypography.scaled(9)
            let title = label.mutableCopy() as! CPTMutableTextStyle
            title.fontName = "SFProText-Medium"
            title.fontSize = NovaTypography.scaled(9.5)

            if let x = axes.xAxis {
                x.title = xAxisTitle
                x.titleOffset = 26
                x.titleTextStyle = title
                x.labelTextStyle = label
                x.axisLineStyle = axisLine
                x.majorTickLineStyle = axisLine
                x.minorTickLineStyle = nil
                x.majorGridLineStyle = gridLine
                x.majorIntervalLength = NSNumber(value: niceInterval(xSpan / 6))
                x.minorTicksPerInterval = 0
                x.axisConstraints = CPTConstraints(lowerOffset: 0)
            }
            if let y = axes.yAxis {
                y.title = yAxisTitle
                y.titleOffset = 42
                y.titleTextStyle = title
                y.labelTextStyle = label
                y.axisLineStyle = axisLine
                y.majorTickLineStyle = axisLine
                y.minorTickLineStyle = nil
                y.majorGridLineStyle = gridLine
                y.majorIntervalLength = NSNumber(value: niceInterval(ySpan / 5))
                y.minorTicksPerInterval = 0
                y.axisConstraints = CPTConstraints(lowerOffset: 0)
                let formatter = NumberFormatter()
                formatter.minimumFractionDigits = 3
                formatter.maximumFractionDigits = 5
                y.labelFormatter = formatter
            }
        }

        private func borderStyle(width: CGFloat, alpha: CGFloat) -> CPTLineStyle {
            let style = CPTMutableLineStyle()
            style.lineWidth = width
            style.lineColor = CPTColor(componentRed: 0.22, green: 0.29, blue: 0.38, alpha: alpha)
            return style
        }

        private func percentile(_ sorted: [Double], fraction: Double) -> Double? {
            guard !sorted.isEmpty else { return nil }
            let index = min(max(Int(Double(sorted.count - 1) * fraction), 0), sorted.count - 1)
            return sorted[index]
        }

        private func niceInterval(_ raw: Double) -> Double {
            guard raw.isFinite, raw > 0 else { return 1 }
            let magnitude = pow(10, floor(log10(raw)))
            let normalized = raw / magnitude
            let nice = normalized < 1.5 ? 1.0 : normalized < 3.5 ? 2.0 : normalized < 7.5 ? 5.0 : 10.0
            return nice * magnitude
        }
    }
}
