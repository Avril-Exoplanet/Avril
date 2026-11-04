import Foundation
import SwiftUI

/// Persistent scientific light-curve workspace shown inside the detection workbench.
///
/// Controls remain mounted while data is being acquired. This avoids the former
/// layout jump and makes the inspection tools discoverable before the first FITS
/// sector reaches the local cache.
struct LightCurveCard: View {
    @EnvironmentObject private var model: AppModel
    let payload: LightCurvePayload?
    let target: TargetRecord?
    @Binding var mode: Int

    @State private var selectedSector = 0
    @State private var displayMode = 0
    @State private var fluxMode = 0
    @State private var phasePeriodDays = 10.0
    @State private var phaseEpochBTJD = 0.0
    @State private var showDataGaps = true
    @State private var resetToken = 0

    private var filteredPoints: [LightCurvePoint] {
        guard let payload else { return [] }
        let source = selectedSector == 0
            ? payload.flattenedPoints
            : payload.flattenedPoints.filter { $0.sector == selectedSector }

        guard displayMode == 1, phasePeriodDays.isFinite, phasePeriodDays > 0 else {
            return source
        }
        return source.map { point in
            var phase = ((point.time - phaseEpochBTJD) / phasePeriodDays)
                .truncatingRemainder(dividingBy: 1)
            if phase < -0.5 { phase += 1 }
            if phase >= 0.5 { phase -= 1 }
            return LightCurvePoint(time: phase, flux: point.flux, sector: point.sector)
        }
        .sorted { $0.time < $1.time }
    }

    private var visibleGaps: [LightCurveGap] {
        guard displayMode == 0, showDataGaps else { return [] }
        return payload?.gaps ?? []
    }

    private var displayPoints: [LightCurvePoint] {
        guard fluxMode == 1 else { return filteredPoints }
        return filteredPoints.map {
            LightCurvePoint(time: $0.time, flux: ($0.flux - 1) * 1_000_000, sector: $0.sector)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            productHeader
                .frame(height: 51)

            Divider().overlay(NovaTheme.border)
            controlDeck
            Divider().overlay(NovaTheme.border)

            if let payload, !filteredPoints.isEmpty {
                plotStatusBar(payload)
                CorePlotLightCurveView(
                    points: displayPoints,
                    gaps: visibleGaps,
                    xAxisTitle: displayMode == 0
                        ? (payload.timeLabel ?? "TIME (BTJD)").uppercased()
                        : "ORBITAL PHASE",
                    yAxisTitle: fluxMode == 0 ? "NORMALIZED FLUX" : "RELATIVE FLUX (PPM)",
                    resetToken: resetToken
                )
                .padding(.horizontal, 7)
                .padding(.top, 4)
                .frame(maxHeight: .infinity)
                .layoutPriority(1)
                .overlay {
                    if let task = liveDisplayTask {
                        LiveSearchPlotOverlay(task: task)
                            .padding(.horizontal, 7)
                            .padding(.top, 4)
                    }
                }

                metricsBar(payload)
            } else {
                LightCurveWaitingPlot(
                    target: target?.title,
                    title: target == nil
                        ? model.t("SELECT A TARGET", "请选择目标")
                        : model.t("AWAITING VERIFIED FITS SAMPLES", "等待验证后的 FITS 样本"),
                    message: target == nil
                        ? model.t("Choose a prioritized TIC target to configure a direct NASA data run.", "选择优先 TIC 目标，配置 NASA 数据直接运行。")
                        : model.t("Start Avril V3. Quality-clean sectors will stream into this time-domain view during acquisition.", "启动 Avril V3。质量清洗后的区段会在获取期间流式显示到这里。")
                )
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(NovaTheme.surface)
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous)
                .stroke(NovaTheme.border, lineWidth: 1)
        }
        .shadow(color: NovaTheme.panelShadow, radius: 12, y: 4)
        .onChange(of: payload?.sectors ?? []) { _, sectors in
            if selectedSector != 0, !sectors.contains(selectedSector) {
                selectedSector = 0
            }
        }
    }

    private var productHeader: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.t("TESS LIGHT-CURVE INSPECTOR", "TESS 光变检查器"))
                    .font(.nova(size: 9, weight: .bold))
                    .tracking(1.05)
                    .foregroundStyle(NovaTheme.accent)
                HStack(spacing: 8) {
                    Text(target?.title ?? model.t("No target selected", "未选择目标"))
                        .font(.nova(size: 13.5, weight: .semibold))
                        .foregroundStyle(NovaTheme.ink)
                    Text(payload?.product ?? "PDCSAP / SAP")
                        .font(.nova(size: 9.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(NovaTheme.inkTertiary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 10)
            SegmentedControl(
                $mode,
                options: [model.t("Orbit", "轨道"), model.t("Light Curve", "光变曲线")],
                prominent: true
            )
            .frame(width: 188)
        }
        .padding(.horizontal, 14)
    }

    private var controlDeck: some View {
        VStack(spacing: 5) {
            HStack(spacing: 9) {
                controlLabel(model.t("SECTOR", "区段"))
                Picker("", selection: $selectedSector) {
                    Text(model.t("All sectors", "全部区段")).tag(0)
                    ForEach(payload?.sectors ?? [], id: \.self) { sector in
                        Text("Sector \(sector)").tag(sector)
                    }
                }
                .labelsHidden()
                .frame(width: 118)
                .disabled(payload == nil)

                controlLabel(model.t("VIEW", "视图"))
                SegmentedControl(
                    $displayMode,
                    options: [model.t("Time", "时间"), model.t("Phase", "相位")]
                )
                .frame(width: 126)

                Toggle(model.t("Mark data gaps", "标记数据缺口"), isOn: $showDataGaps)
                    .toggleStyle(.checkbox)
                    .font(.nova(size: 10.5, weight: .medium))
                    .foregroundStyle(NovaTheme.inkSecondary)
                    .disabled(displayMode == 1 || payload == nil)

                Spacer(minLength: 8)

                Button {
                    resetToken &+= 1
                } label: {
                    Label(model.t("Reset View", "重置视图"), systemImage: "arrow.counterclockwise")
                }
                .buttonStyle(SecondaryButtonStyle(compact: true))
                .disabled(payload == nil)
            }

            HStack(spacing: 8) {
                if displayMode == 1 {
                    controlLabel(model.t("PERIOD", "周期"))
                    TextField("", value: $phasePeriodDays, format: .number.precision(.fractionLength(5)))
                        .textFieldStyle(.roundedBorder)
                        .font(.nova(size: 10.5, design: .monospaced))
                        .frame(width: 86)
                    Text(model.t("days", "天"))
                        .font(.nova(size: 9.5))
                        .foregroundStyle(NovaTheme.inkTertiary)
                    controlLabel(model.t("EPOCH", "历元"))
                    TextField("", value: $phaseEpochBTJD, format: .number.precision(.fractionLength(5)))
                        .textFieldStyle(.roundedBorder)
                        .font(.nova(size: 10.5, design: .monospaced))
                        .frame(width: 92)
                    Text("BTJD")
                        .font(.nova(size: 9.5, design: .monospaced))
                        .foregroundStyle(NovaTheme.inkTertiary)
                    Spacer()
                    Text(model.t("Display-only fold; Avril search remains unchanged", "仅改变显示；不会改动 Avril 搜索"))
                        .font(.nova(size: 9.5))
                        .foregroundStyle(NovaTheme.inkTertiary)
                } else {
                    controlLabel(model.t("FLUX", "通量"))
                    SegmentedControl(
                        $fluxMode,
                        options: [model.t("Normalized", "归一化"), "ppm"]
                    )
                    .frame(width: 132)
                    Image(systemName: "cursorarrow.motionlines")
                        .foregroundStyle(NovaTheme.accent)
                    Text(model.t(
                        "Drag to pan · Scroll at cursor to zoom · Shift-scroll for Y · Double-click to reset",
                        "拖动平移 · 在光标处滚轮缩放 · Shift+滚轮缩放 Y · 双击重置"
                    ))
                    .font(.nova(size: 9.5, weight: .medium))
                    .foregroundStyle(NovaTheme.inkSecondary)
                    .lineLimit(1)
                    Spacer(minLength: 8)
                    Text(model.t("QUALITY-CLEAN · MEDIAN-NORMALIZED · EXTREMA-PRESERVED", "质量清洗 · 中位数归一化 · 极值保留"))
                        .font(.nova(size: 8.5, weight: .semibold, design: .monospaced))
                        .foregroundStyle(NovaTheme.inkTertiary)
                        .lineLimit(1)
                }
            }
            .frame(height: 24)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(NovaTheme.surfaceStrong)
    }

    private func metricsBar(_ payload: LightCurvePayload) -> some View {
        HStack(spacing: 15) {
            scienceDatum(
                model.t("SECTORS", "区段"),
                selectedSector == 0 ? "\(payload.sectors.count)" : "S\(selectedSector)"
            )
            scienceDatum(model.t("VISIBLE", "可见点数"), filteredPoints.count.formatted())
            scienceDatum(model.t("VALID SAMPLES", "有效样本"), payload.validSamples?.formatted() ?? "—")
            scienceDatum(model.t("CADENCE", "采样间隔"), cadenceText(payload.cadenceMedianDays))
            scienceDatum(model.t("BASELINE", "时间跨度"), payload.baselineDays.map { "\($0.oneDecimal) d" } ?? "—")
            scienceDatum(model.t("ROBUST SCATTER", "稳健散布"), payload.robustScatterPPM.map { "\($0.oneDecimal) ppm" } ?? "—")
            scienceDatum(model.t("GAPS", "缺口"), "\(payload.gaps.count)")
            Spacer(minLength: 2)
        }
        .padding(.horizontal, 14)
        .frame(height: 39)
        .background(NovaTheme.surfaceStrong)
        .overlay(alignment: .top) { Divider().overlay(NovaTheme.border) }
    }

    private func plotStatusBar(_ payload: LightCurvePayload) -> some View {
        HStack(spacing: 10) {
            Text(displayMode == 0 ? model.t("TIME DOMAIN", "时间域") : model.t("PHASE DOMAIN", "相位域"))
                .font(.nova(size: 8.5, weight: .bold))
                .tracking(0.9)
                .foregroundStyle(NovaTheme.accent)
            Rectangle().fill(NovaTheme.borderStrong).frame(width: 1, height: 12)
            StatusPill(label: model.t("QUALITY = 0", "质量标记 = 0"), tint: NovaTheme.green)
            Text(model.t("Extrema-preserving display decimation", "保留极值的显示抽样"))
                .font(.nova(size: 9, weight: .medium))
                .foregroundStyle(NovaTheme.inkTertiary)
            Spacer()
            ForEach(Array(payload.sectors.prefix(5)), id: \.self) { sector in
                Text("S\(sector)")
                    .font(.nova(size: 8.5, weight: .semibold, design: .monospaced))
                    .foregroundStyle(NovaTheme.inkSecondary)
            }
            if payload.sectors.count > 5 {
                Text("+\(payload.sectors.count - 5)")
                    .font(.nova(size: 8.5, weight: .semibold, design: .monospaced))
                    .foregroundStyle(NovaTheme.inkTertiary)
            }
        }
        .padding(.horizontal, 12)
        .frame(height: 27)
        .background(NovaTheme.surface)
    }

    private func controlLabel(_ text: String) -> some View {
        Text(text)
            .font(.nova(size: 8.5, weight: .bold))
            .tracking(0.55)
            .foregroundStyle(NovaTheme.inkTertiary)
    }

    private func scienceDatum(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.nova(size: 7.5, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(NovaTheme.inkTertiary)
            Text(value)
                .font(.nova(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(NovaTheme.ink)
                .lineLimit(1)
        }
    }

    private var liveDisplayTask: ScienceTask? {
        guard let task = model.activeTask,
              model.lightCurveTaskID == task.id,
              task.state == .running else { return nil }
        return task
    }

    private func cadenceText(_ cadenceDays: Double?) -> String {
        guard let cadenceDays, cadenceDays.isFinite, cadenceDays > 0 else { return "—" }
        let seconds = cadenceDays * 86_400
        return seconds >= 120
            ? String(format: "%.1f min", seconds / 60)
            : String(format: "%.0f s", seconds)
    }
}

/// A deliberately lightweight analysis animation. It communicates that the
/// plotted samples are being consumed by the current real pipeline stage; it
/// never draws fabricated transit markers or candidate ephemerides.
private struct LiveSearchPlotOverlay: View {
    let task: ScienceTask
    @State private var scanning = false

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .topLeading) {
                LinearGradient(
                    colors: [.clear, NovaTheme.accent.opacity(0.04), NovaTheme.accent.opacity(0.18), .clear],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .frame(width: 86)
                .offset(x: scanning ? proxy.size.width + 86 : -86)
                .animation(.linear(duration: 3.0).repeatForever(autoreverses: false), value: scanning)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 7) {
                        ProgressView().controlSize(.mini).tint(NovaTheme.accent)
                        Text(task.stage.replacingOccurrences(of: "_", with: " ").uppercased())
                            .font(.nova(size: 8.5, weight: .bold, design: .monospaced))
                            .tracking(0.7)
                            .foregroundStyle(NovaTheme.accent)
                        Text("\(Int((task.progress * 100).rounded()))%")
                            .font(.nova(size: 8.5, weight: .semibold, design: .monospaced))
                            .foregroundStyle(NovaTheme.inkSecondary)
                    }
                    Text(task.message)
                        .font(.nova(size: 9))
                        .foregroundStyle(NovaTheme.inkSecondary)
                        .lineLimit(1)
                }
                .padding(.horizontal, 9)
                .padding(.vertical, 7)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 5).stroke(NovaTheme.border, lineWidth: 1))
                .padding(9)
            }
            .clipped()
            .onAppear { scanning = true }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// A real empty state that still reads as an instrument viewport. It draws
/// axes and a measurement grid but never invents science samples.
private struct LightCurveWaitingPlot: View {
    let target: String?
    let title: String
    let message: String

    var body: some View {
        ZStack {
            Canvas { context, size in
                let rect = CGRect(x: 58, y: 14, width: max(size.width - 76, 1), height: max(size.height - 52, 1))
                var grid = Path()
                for index in 0...8 {
                    let x = rect.minX + rect.width * CGFloat(index) / 8
                    grid.move(to: CGPoint(x: x, y: rect.minY))
                    grid.addLine(to: CGPoint(x: x, y: rect.maxY))
                }
                for index in 0...5 {
                    let y = rect.minY + rect.height * CGFloat(index) / 5
                    grid.move(to: CGPoint(x: rect.minX, y: y))
                    grid.addLine(to: CGPoint(x: rect.maxX, y: y))
                }
                context.stroke(grid, with: .color(NovaTheme.border.opacity(0.68)), lineWidth: 0.65)

                var axes = Path()
                axes.move(to: CGPoint(x: rect.minX, y: rect.minY))
                axes.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
                axes.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
                context.stroke(axes, with: .color(NovaTheme.borderStrong), lineWidth: 0.9)
            }

            VStack(spacing: 6) {
                HStack(spacing: 7) {
                    Circle().fill(NovaTheme.amber).frame(width: 6, height: 6)
                    Text(title)
                        .font(.nova(size: 10, weight: .bold, design: .monospaced))
                        .tracking(0.55)
                }
                .foregroundStyle(NovaTheme.ink)
                if let target {
                    Text(target)
                        .font(.nova(size: 13, weight: .semibold, design: .monospaced))
                        .foregroundStyle(NovaTheme.accent)
                }
                Text(message)
                    .font(.nova(size: 10.5))
                    .foregroundStyle(NovaTheme.inkSecondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 14)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(NovaTheme.borderStrong, lineWidth: 1))
        }
        .background(NovaTheme.surfaceMuted)
        .overlay(alignment: .bottom) {
            Text("TIME (BTJD)")
                .font(.nova(size: 8.5, weight: .semibold, design: .monospaced))
                .foregroundStyle(NovaTheme.inkTertiary)
                .padding(.bottom, 5)
        }
        .overlay(alignment: .leading) {
            Text("NORMALIZED FLUX")
                .font(.nova(size: 8.5, weight: .semibold, design: .monospaced))
                .foregroundStyle(NovaTheme.inkTertiary)
                .rotationEffect(.degrees(-90))
                .offset(x: -23)
        }
        .padding(7)
    }
}
