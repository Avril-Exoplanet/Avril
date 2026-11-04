import Foundation
import SwiftUI

/// A lossless projection of Avril's event ledger into the handful of values a
/// scientist needs while a target is running. Unknown values remain nil; the
/// UI never invents a peak, family or candidate count.
struct PipelineTelemetry {
    let loadedSectors: Int
    let expectedSectors: Int?
    let validSamples: Int?
    let completedSearchBlocks: Int?
    let totalSearchBlocks: Int?
    let rawPeaks: Int?
    let voteSeedsComplete: Int?
    let voteSeedsTotal: Int?
    let promotedFamilies: Int?
    let refitsComplete: Int?
    let refitsTotal: Int?
    let refitsRescued: Int?
    let stableFamilies: Int?
    let eventChecksComplete: Int?
    let eventChecksTotal: Int?
    let uniqueEventTrains: Int?
    let meaningfulSignals: Int?
    let gapDominated: Int?
    let eventIncoherent: Int?
    let reviewedComplete: Int?
    let reviewedTotal: Int?
    let phaseNullComplete: Int?
    let phaseNullTotal: Int?
    let manualReviewRequired: Int?
    let lowPriorityOrFalseAlarmRisk: Int?
    let passes7p1Reference: Int?
    let auditRows: Int?
    let stageIndex: Int?

    init(task: ScienceTask?, payload: LightCurvePayload?, target: TargetRecord?) {
        let events = task?.events.sorted { ($0.seq ?? 0) < ($1.seq ?? 0) } ?? []
        loadedSectors = payload?.sectors.count ?? 0
        expectedSectors = target?.sectorCount
        validSamples = payload?.validSamples

        let searchEvents = events.filter { Self.normalized($0.stage) == "searching" }
        let blockNumbers = Set(searchEvents.compactMap { event in
            event.metrics?.block ?? Self.fraction(in: event.message)?.current
        })
        completedSearchBlocks = blockNumbers.isEmpty ? nil : blockNumbers.count
        totalSearchBlocks = searchEvents.compactMap { Self.fraction(in: $0.message)?.total }.max()

        rawPeaks = Self.latest(events) { $0.rawHypotheses }

        let initialVoteEvents = events.filter {
            Self.normalized($0.stage) == "multi_seed_voting" &&
            !($0.message ?? "").localizedCaseInsensitiveContains("changed-ephemeris")
        }
        voteSeedsComplete = Self.latest(initialVoteEvents) { $0.seedIndex }
            ?? initialVoteEvents.compactMap { Self.fraction(in: $0.message)?.current }.max()
        voteSeedsTotal = Self.latest(initialVoteEvents) { $0.seedCount }
            ?? initialVoteEvents.compactMap { Self.fraction(in: $0.message)?.total }.max()

        promotedFamilies = Self.latest(events) { $0.promotedFamilies }
        refitsComplete = Self.latest(events) { $0.familiesCompleted }
            ?? events.filter { Self.normalized($0.stage) == "refitting" }
                .compactMap { Self.fraction(in: $0.message)?.current }.max()
        refitsTotal = Self.latest(events) { $0.familiesTotal }
            ?? events.filter { Self.normalized($0.stage) == "refitting" }
                .compactMap { Self.fraction(in: $0.message)?.total }.max()
        refitsRescued = Self.latest(events) { $0.familiesRescued }
        stableFamilies = Self.latest(events) { $0.stableFamilies }

        let eventRows = events.filter { Self.normalized($0.stage) == "event_repeatability" }
        eventChecksComplete = Self.latest(eventRows) { $0.familiesCompleted }
            ?? eventRows.compactMap { Self.fraction(in: $0.message)?.current }.max()
        eventChecksTotal = Self.latest(eventRows) { $0.familiesTotal }
            ?? eventRows.compactMap { Self.fraction(in: $0.message)?.total }.max()
        uniqueEventTrains = Self.latest(events) { $0.deduplicatedFamilies }
            ?? Self.latest(events) { $0.deduplicatedCandidates }

        let latestFunnel = Self.latestFunnel(events)
        meaningfulSignals = latestFunnel?.meaningfulCandidates
            ?? Self.latest(events) { $0.eligible }
        gapDominated = latestFunnel?.dispositionCounts?.gapDominated
        eventIncoherent = latestFunnel?.dispositionCounts?.eventIncoherent

        let reviewRows = events.filter { Self.normalized($0.stage) == "vetting" }
        reviewedComplete = Self.latest(reviewRows) { $0.candidatesCompleted }
            ?? reviewRows.compactMap { Self.fraction(in: $0.message)?.current }.max()
        reviewedTotal = Self.latest(reviewRows) { $0.candidatesTotal }
            ?? reviewRows.compactMap { Self.fraction(in: $0.message)?.total }.max()

        let significanceRows = events.filter { Self.normalized($0.stage) == "significance" }
        phaseNullComplete = Self.latest(significanceRows) { $0.candidate }
            ?? significanceRows.compactMap { Self.fraction(in: $0.message)?.current }.max()
        phaseNullTotal = Self.latest(significanceRows) { $0.total }
            ?? significanceRows.compactMap { Self.fraction(in: $0.message)?.total }.max()

        manualReviewRequired = Self.latest(events) { $0.manualReviewRequired }
        lowPriorityOrFalseAlarmRisk = Self.latest(events) { $0.lowPriorityOrFalseAlarmRisk }
        passes7p1Reference = Self.latest(events) { $0.passes7p1Reference }

        auditRows = latestFunnel?.inputCandidates
            ?? Self.latest(events) { $0.allCandidates }
            ?? task?.candidateCount
        stageIndex = Self.pipelineStageIndex(task: task, events: events)
    }

    private static func latest(
        _ events: [JobEvent],
        _ value: (JobEventMetrics) -> Int?
    ) -> Int? {
        for event in events.reversed() {
            if let metrics = event.metrics, let result = value(metrics) { return result }
        }
        return nil
    }

    private static func latestFunnel(_ events: [JobEvent]) -> MeaningfulCandidateFunnel? {
        events.reversed().compactMap { $0.metrics?.meaningfulCandidateFunnel }.first
    }

    private static func normalized(_ stage: String?) -> String {
        (stage ?? "").split(separator: ":").last.map(String.init)?.lowercased() ?? ""
    }

    private static func fraction(in message: String?) -> (current: Int, total: Int)? {
        guard let message else { return nil }
        for token in message.split(whereSeparator: \.isWhitespace) {
            let clean = token.trimmingCharacters(in: .punctuationCharacters)
            let parts = clean.split(separator: "/", maxSplits: 1)
            if parts.count == 2, let current = Int(parts[0]), let total = Int(parts[1]) {
                return (current, total)
            }
        }
        return nil
    }

    private static func pipelineStageIndex(task: ScienceTask?, events: [JobEvent]) -> Int? {
        if task?.state == .complete { return 6 }
        guard let latest = events.last else {
            return task?.state == .running ? 0 : nil
        }
        switch normalized(latest.stage) {
        case "queued", "starting", "source", "discovering", "download", "downloading", "launch", "target_start", "loading", "preparing":
            return 0
        case "searching":
            return 1
        case "voting", "multi_seed_voting", "matching", "validating":
            return 2
        case "refitting", "revoting", "discovery_complete":
            return 3
        case "event_repeatability", "deduplicating", "harmonics", "harmonic_families":
            return 4
        case "physical_reference_gate", "vetting", "pixel_vetting", "exact_phase_null_gate", "significance":
            return 5
        case "complete", "target_complete", "rendering":
            return 6
        default:
            return task?.state == .running ? 0 : nil
        }
    }
}

struct RunStageRibbon: View {
    @EnvironmentObject private var model: AppModel
    let task: ScienceTask?
    let telemetry: PipelineTelemetry

    private struct Stage: Identifiable {
        let id: Int
        let title: String
        let detail: String
        let symbol: String
    }

    var body: some View {
        HStack(spacing: 0) {
            ForEach(stages) { stage in
                stageCell(stage)
                if stage.id < stages.count - 1 {
                    Rectangle()
                        .fill(connectorColor(after: stage.id))
                        .frame(width: 18, height: 1)
                }
            }
        }
        .padding(.horizontal, 15)
        .padding(.vertical, 11)
        .background(
            LinearGradient(
                colors: [Color(hex: 0x111A23), Color(hex: 0x16232D)],
                startPoint: .leading,
                endPoint: .trailing
            )
        )
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: NovaTheme.panelRadius)
                .stroke(Color.white.opacity(0.10), lineWidth: 1)
        }
        .shadow(color: Color.black.opacity(0.08), radius: 8, y: 3)
    }

    private var stages: [Stage] {
        [
            Stage(id: 0, title: model.t("Acquire", "数据获取"), detail: sectorDetail, symbol: "arrow.down.circle"),
            Stage(id: 1, title: model.t("Search", "全域搜索"), detail: searchDetail, symbol: "waveform.path.ecg"),
            Stage(id: 2, title: model.t("Consensus", "交叉投票"), detail: consensusDetail, symbol: "point.3.connected.trianglepath.dotted"),
            Stage(id: 3, title: model.t("Refit", "精密重拟合"), detail: refitDetail, symbol: "scope"),
            Stage(id: 4, title: model.t("Events", "事件一致性"), detail: eventDetail, symbol: "timeline.selection"),
            Stage(id: 5, title: model.t("Vet", "科学排假"), detail: vetDetail, symbol: "checkmark.shield"),
            Stage(id: 6, title: model.t("Report", "审计报告"), detail: reportDetail, symbol: "doc.text.magnifyingglass"),
        ]
    }

    private var sectorDetail: String {
        let loaded = telemetry.loadedSectors
        let total = telemetry.expectedSectors.map(String.init) ?? "—"
        return model.t("\(loaded)/\(total) sectors", "\(loaded)/\(total) 个区段")
    }

    private var searchDetail: String {
        if let done = telemetry.completedSearchBlocks, let total = telemetry.totalSearchBlocks {
            return model.t("\(done)/\(total) blocks", "\(done)/\(total) 个时间块")
        }
        return model.t("Awaiting blocks", "等待时间块")
    }

    private var consensusDetail: String {
        if let done = telemetry.voteSeedsComplete, let total = telemetry.voteSeedsTotal {
            return model.t("\(done)/\(total) vote seeds", "\(done)/\(total) 组投票")
        }
        return model.t("Awaiting peaks", "等待搜索峰")
    }

    private var refitDetail: String {
        fraction(telemetry.refitsComplete, telemetry.refitsTotal, english: "families", chinese: "候选族")
    }

    private var eventDetail: String {
        fraction(telemetry.eventChecksComplete, telemetry.eventChecksTotal, english: "event trains", chinese: "事件序列")
    }

    private var vetDetail: String {
        fraction(telemetry.reviewedComplete, telemetry.reviewedTotal, english: "reviewed", chinese: "已复核")
    }

    private var reportDetail: String {
        telemetry.auditRows.map { model.t("\($0.formatted()) audit rows", "\($0.formatted()) 条审计记录") }
            ?? model.t("Awaiting ledger", "等待证据账本")
    }

    private func fraction(_ current: Int?, _ total: Int?, english: String, chinese: String) -> String {
        guard let current, let total else { return model.t("Awaiting data", "等待数据") }
        return model.t("\(current)/\(total) \(english)", "\(current)/\(total) \(chinese)")
    }

    private func stageCell(_ stage: Stage) -> some View {
        let current = telemetry.stageIndex
        let isDone = task?.state == .complete || (current.map { stage.id < $0 } ?? false)
        let isActive = task?.state == .running && current == stage.id
        let tint = isDone ? NovaTheme.green : isActive ? Color(hex: 0x68D8FF) : Color.white.opacity(0.34)

        return HStack(spacing: 8) {
            ZStack {
                Circle()
                    .fill(tint.opacity(isActive ? 0.18 : 0.10))
                    .frame(width: 29, height: 29)
                if isDone {
                    Image(systemName: "checkmark")
                } else if isActive {
                    ProgressView().controlSize(.mini).tint(tint)
                } else {
                    Image(systemName: stage.symbol)
                }
            }
            .font(.nova(size: 10, weight: .semibold))
            .foregroundStyle(tint)

            VStack(alignment: .leading, spacing: 2) {
                Text(stage.title.uppercased())
                    .font(.nova(size: 9, weight: .bold))
                    .tracking(0.7)
                    .foregroundStyle(isActive || isDone ? Color.white : Color.white.opacity(0.52))
                Text(stage.detail)
                    .font(.nova(size: 8.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(isActive ? Color(hex: 0x9DE8FF) : Color.white.opacity(0.42))
                    .lineLimit(1)
            }
            Spacer(minLength: 3)
        }
        .frame(maxWidth: .infinity, minHeight: 37, alignment: .leading)
        .padding(.horizontal, 7)
        .background(isActive ? Color(hex: 0x1F7797).opacity(0.18) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .animation(.easeInOut(duration: 0.28), value: current)
    }

    private func connectorColor(after index: Int) -> Color {
        guard let current = telemetry.stageIndex else { return Color.white.opacity(0.10) }
        return index < current ? NovaTheme.green.opacity(0.75) : Color.white.opacity(0.12)
    }
}

struct LiveTelemetryPanel: View {
    @EnvironmentObject private var model: AppModel
    let task: ScienceTask?
    let target: TargetRecord?
    let telemetry: PipelineTelemetry

    private let columns = [GridItem(.flexible()), GridItem(.flexible())]

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            statusHeader
            ProgressView(value: task?.progress ?? 0)
                .tint(task?.state == .complete ? NovaTheme.green : NovaTheme.accent)

            LazyVGrid(columns: columns, spacing: 7) {
                metric(model.t("SECTORS", "观测区段"), sectorValue, note: sampleNote)
                metric(model.t("RAW PEAKS", "原始搜索峰"), number(telemetry.rawPeaks), note: blockNote)
                metric(model.t("PROMOTED", "晋级候选族"), number(telemetry.promotedFamilies), note: voteNote)
                metric(model.t("EXACT REFITS", "精密重拟合"), fraction(telemetry.refitsComplete, telemetry.refitsTotal), note: rescuedNote)
                metric(model.t("STABLE FAMILIES", "稳定候选族"), number(telemetry.stableFamilies), note: eventNote)
                metric(model.t("UNIQUE TRAINS", "独立事件序列"), number(telemetry.uniqueEventTrains), note: model.t("after alias collapse", "折叠别名后"))
                metric(model.t("DIAGNOSTIC SET", "诊断集"), fraction(telemetry.reviewedComplete, telemetry.reviewedTotal), note: diagnosticNote)
                metric(model.t("AUDIT LEDGER", "审计记录"), number(telemetry.auditRows), note: outcomeNote)
            }

            currentActivity
            Spacer(minLength: 0)
            runControls
        }
        .padding(10)
        .background(NovaTheme.surface)
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.panelRadius).stroke(NovaTheme.border, lineWidth: 1))
        .shadow(color: NovaTheme.panelShadow, radius: 9, y: 3)
    }

    private var statusHeader: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.t("LIVE RUN OBSERVATORY", "实时运行观测台"))
                        .font(.nova(size: 9, weight: .bold))
                        .tracking(1.0)
                        .foregroundStyle(NovaTheme.accent)
                    Text(target?.title ?? model.t("No target selected", "未选择目标"))
                        .font(.nova(size: 18, weight: .semibold, design: .rounded))
                        .foregroundStyle(NovaTheme.ink)
                }
                Spacer()
                StatusPill(label: stateLabel, tint: task?.state.tint ?? NovaTheme.inkTertiary)
            }
            Text(statusMessage)
                .font(.nova(size: 9.5))
                .foregroundStyle(NovaTheme.inkSecondary)
                .lineLimit(1)
                .frame(minHeight: 16, alignment: .topLeading)
        }
    }

    private var currentActivity: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(model.t("LIVE EVENT TRACE", "实时事件轨迹"))
                    .font(.nova(size: 8.5, weight: .bold))
                    .tracking(0.8)
                    .foregroundStyle(NovaTheme.inkTertiary)
                Spacer()
                if task?.state == .running { ProgressView().controlSize(.mini) }
            }
            ForEach(Array(recentEvents.enumerated()), id: \.offset) { _, event in
                HStack(alignment: .firstTextBaseline, spacing: 7) {
                    Text(String(format: "%03d", event.seq ?? 0))
                        .font(.nova(size: 8.5, design: .monospaced))
                        .foregroundStyle(NovaTheme.accent)
                        .frame(width: 25, alignment: .trailing)
                    Text(shortStageLabel(event.stage))
                        .font(.nova(size: 8, weight: .bold, design: .monospaced))
                        .foregroundStyle(NovaTheme.inkTertiary)
                        .frame(width: 82, alignment: .leading)
                    Text(localizedEventMessage(event))
                        .font(.nova(size: 9))
                        .foregroundStyle(NovaTheme.inkSecondary)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
            }
        }
        .padding(6)
        .background(NovaTheme.surfaceMuted)
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5).stroke(NovaTheme.border, lineWidth: 1))
    }

    private var runControls: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Picker(model.t("Data source", "数据源"), selection: $model.source) {
                    Text(model.t("Auto source", "自动数据源")).tag("auto")
                    Text("SPOC").tag("spoc")
                    Text("FFI").tag("ffi")
                }
                .labelsHidden()
                .controlSize(.small)
                .frame(maxWidth: .infinity)
                .disabled(task?.state == .running)

                Toggle(model.t("Pixels", "像素"), isOn: $model.includePixelFollowup)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .font(.nova(size: 9.5, weight: .medium))
                    .disabled(task?.state == .running)
            }
            HStack(spacing: 7) {
                Button { model.addSelectedToQueue(runImmediately: true) } label: {
                    Label(model.t("Run selected target", "运行选定目标"), systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(PrimaryButtonStyle(compact: true))
                .disabled(model.selectedTarget == nil || task?.state == .running)

                if let task, [.queued, .running].contains(task.state) {
                    Button { model.stop(task: task) } label: {
                        Image(systemName: "stop.fill")
                    }
                    .buttonStyle(SecondaryButtonStyle(compact: true))
                    .foregroundStyle(NovaTheme.red)
                } else {
                    Button { model.addSelectedToQueue() } label: {
                        Image(systemName: "text.badge.plus")
                    }
                    .buttonStyle(SecondaryButtonStyle(compact: true))
                    .disabled(model.selectedTarget == nil)
                }
            }
        }
    }

    private func metric(_ label: String, _ value: String, note: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.nova(size: 7.5, weight: .bold)).tracking(0.55).foregroundStyle(NovaTheme.inkTertiary)
            Text(value).font(.nova(size: 15, weight: .semibold, design: .monospaced)).foregroundStyle(NovaTheme.ink).monospacedDigit()
            Text(note).font(.nova(size: 8.5)).foregroundStyle(NovaTheme.inkSecondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
        .padding(.horizontal, 9)
        .padding(.vertical, 3)
        .background(NovaTheme.surfaceMuted)
        .clipShape(RoundedRectangle(cornerRadius: 5))
        .overlay(RoundedRectangle(cornerRadius: 5).stroke(NovaTheme.border, lineWidth: 1))
        .contentTransition(.numericText())
    }

    private var stateLabel: String {
        guard let task else { return model.t("STANDBY", "待命") }
        switch task.state {
        case .queued: return model.t("QUEUED", "排队")
        case .running: return model.t("RUNNING", "运行中")
        case .complete: return model.t("COMPLETE", "已完成")
        case .cancelled: return model.t("STOPPED", "已停止")
        case .failed: return model.t("FAILED", "失败")
        }
    }

    private var sectorValue: String {
        let total = telemetry.expectedSectors.map(String.init) ?? "—"
        return "\(telemetry.loadedSectors) / \(total)"
    }

    private var sampleNote: String {
        telemetry.validSamples.map { model.t("\($0.formatted()) valid samples", "\($0.formatted()) 个有效采样") }
            ?? model.t("waiting for FITS", "等待 FITS")
    }

    private var blockNote: String {
        guard let done = telemetry.completedSearchBlocks, let total = telemetry.totalSearchBlocks else {
            return model.t("search not started", "搜索尚未开始")
        }
        return model.t("\(done)/\(total) temporal blocks", "\(done)/\(total) 个时间块")
    }

    private var voteNote: String {
        guard let done = telemetry.voteSeedsComplete, let total = telemetry.voteSeedsTotal else {
            return model.t("awaiting consensus", "等待交叉投票")
        }
        return model.t("\(done)/\(total) vote seeds", "\(done)/\(total) 组投票")
    }

    private var rescuedNote: String {
        telemetry.refitsRescued.map { model.t("\($0) rescued", "救回 \($0) 个") }
            ?? model.t("evidence-backed budget", "证据驱动的精度预算")
    }

    private var eventNote: String {
        guard let done = telemetry.eventChecksComplete, let total = telemetry.eventChecksTotal else {
            return model.t("awaiting event checks", "等待事件检查")
        }
        return model.t("\(done)/\(total) event checks", "\(done)/\(total) 事件检查")
    }

    private var diagnosticNote: String {
        telemetry.meaningfulSignals.map {
            model.t("\($0) unverified; not confirmed", "\($0) 条未验证，不代表确认")
        } ?? model.t("non-destructive evidence review", "非破坏性证据复核")
    }

    private var outcomeNote: String {
        if let passes = telemetry.passes7p1Reference {
            return model.t("\(passes) passed the 7.1 reference", "\(passes) 条通过 7.1 参考门槛")
        }
        if let done = telemetry.phaseNullComplete, let total = telemetry.phaseNullTotal {
            return model.t("\(done)/\(total) phase-null tested", "\(done)/\(total) 已做相位零假设")
        }
        return model.t("not a planet count", "不是行星数量")
    }

    private var recentEvents: [JobEvent] {
        let limit = 2
        return Array((task?.events ?? []).filter {
            let message = $0.message ?? ""
            return !message.isEmpty && message != "The science engine reported a new execution step"
        }.suffix(limit))
    }

    private var statusMessage: String {
        guard let task else {
            return model.t(
                "Configure a target, then start the local Avril V3 pipeline.",
                "配置目标，然后启动本地 Avril V3 流程。"
            )
        }
        switch task.state {
        case .complete:
            return model.t(
                "The evidence ledger and audit report are ready.",
                "证据记录与审计报告已生成。"
            )
        case .failed:
            return model.t(task.message, "运行失败，已保留当前记录。")
        case .cancelled:
            return model.t(task.message, "运行已停止，已保留当前记录。")
        case .queued:
            return model.t(task.message, "正在等待本地科学引擎。")
        case .running:
            if let latest = recentEvents.last { return localizedEventMessage(latest) }
            return model.t(task.message, "本地 Avril V3 科学流程正在运行。")
        }
    }

    private func shortStageLabel(_ stage: String?) -> String {
        let normalized = (stage ?? "engine").split(separator: ":").last.map(String.init)?.lowercased() ?? "engine"
        switch normalized {
        case "exact_phase_null_gate": return model.t("PHASE NULL", "相位零检验")
        case "event_repeatability": return model.t("EVENT TEST", "事件复现")
        case "physical_reference_gate": return model.t("SCIENCE GATE", "科学门槛")
        case "multi_seed_voting": return model.t("CONSENSUS", "交叉投票")
        case "discovery_complete": return model.t("DISCOVERY", "搜索完成")
        case "significance": return model.t("SIGNIFICANCE", "显著性")
        case "complete": return model.t("COMPLETE", "报告完成")
        default:
            return normalized.replacingOccurrences(of: "_", with: " ").uppercased()
        }
    }

    private func localizedEventMessage(_ event: JobEvent) -> String {
        let english = event.message ?? event.title ?? "—"
        let normalized = (event.stage ?? "engine").split(separator: ":").last.map(String.init)?.lowercased() ?? "engine"
        let chinese: String
        switch normalized {
        case "exact_phase_null_gate": chinese = "正在用相位平移零假设校准合格星历"
        case "significance": chinese = "正在计算红噪声 MES-like 显著性与虚警率"
        case "complete": chinese = "完整候选证据记录与报告已生成"
        case "vetting": chinese = "正在执行物理诊断与轻量参考检查"
        case "event_repeatability": chinese = "正在检查每个预测事件的跨区段复现性"
        case "refitting": chinese = "正在对候选族执行精密周期与历元重拟合"
        case "searching": chinese = "正在独立时间块上执行全域信号搜索"
        default: chinese = "Avril V3 正在执行当前证据步骤"
        }
        return model.t(english, chinese)
    }

    private func number(_ value: Int?) -> String { value?.formatted() ?? "—" }
    private func fraction(_ current: Int?, _ total: Int?) -> String {
        guard let current, let total else { return "—" }
        return "\(current) / \(total)"
    }
}
