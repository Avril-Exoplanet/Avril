import SwiftUI

struct WorkbenchView: View {
    @EnvironmentObject private var model: AppModel
    @State private var missionMode = 0

    var body: some View {
        GeometryReader { proxy in
            let contentWidth = max(proxy.size.width - 32, 620)
            if proxy.size.width < 1080 || proxy.size.height < 760 {
                compactLayout(contentWidth: contentWidth)
            } else {
                desktopLayout(contentWidth: contentWidth, viewportHeight: proxy.size.height)
            }
        }
        .background(NovaTheme.canvas)
        .onAppear {
            if model.activeTask != nil || model.lightCurve != nil { missionMode = 1 }
        }
        .onChange(of: model.activeTask?.id) { _, taskID in
            if taskID != nil { missionMode = 1 }
        }
        .onChange(of: model.lightCurveTaskID) { _, taskID in
            if taskID != nil { missionMode = 1 }
        }
    }

    private func desktopLayout(contentWidth: CGFloat, viewportHeight: CGFloat) -> some View {
        let missionHeight = min(500, max(410, viewportHeight * 0.48))
        let focusTask = model.focusedRunTask
        let focusTarget = focusTask?.target ?? model.observedTarget ?? model.selectedTarget
        let telemetry = PipelineTelemetry(task: focusTask, payload: model.lightCurve, target: focusTarget)
        return VStack(spacing: 14) {
            header
                .frame(width: contentWidth)

            RunStageRibbon(task: focusTask, telemetry: telemetry)
                .frame(width: contentWidth, height: 64)

            HStack(alignment: .top, spacing: 14) {
                mission(height: missionHeight)
                    .frame(width: max(contentWidth - 394, 540))
                LiveTelemetryPanel(task: focusTask, target: focusTarget, telemetry: telemetry)
                    .frame(width: 380, height: missionHeight)
                    .clipped()
            }
            .frame(width: contentWidth, height: missionHeight, alignment: .leading)

            HStack(alignment: .top, spacing: 14) {
                TargetCatalogView()
                    .frame(width: max(contentWidth - 374, 540))
                    .frame(maxHeight: .infinity)
                TaskQueueCard()
                    .frame(width: 360)
                .frame(maxHeight: .infinity)
            }
            .frame(width: contentWidth, alignment: .topLeading)
            .frame(maxHeight: .infinity, alignment: .top)
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func compactLayout(contentWidth: CGFloat) -> some View {
        let focusTask = model.focusedRunTask
        let focusTarget = focusTask?.target ?? model.observedTarget ?? model.selectedTarget
        let telemetry = PipelineTelemetry(task: focusTask, payload: model.lightCurve, target: focusTarget)
        return ScrollView(.vertical) {
            VStack(spacing: 14) {
                header
                RunStageRibbon(task: focusTask, telemetry: telemetry)
                    .frame(height: 64)
                mission(height: 420)
                LiveTelemetryPanel(task: focusTask, target: focusTarget, telemetry: telemetry)
                    .frame(height: 500)
                TargetCatalogView().frame(height: 450)
                TaskQueueCard()
            }
            .frame(width: contentWidth, alignment: .topLeading)
            .padding(16)
        }
        .scrollIndicators(.visible)
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 20) {
            SectionHeading(
                eyebrow: model.t("Avril V3 discovery system", "Avril V3 发现系统"),
                title: model.t("Detection Workbench", "系外行星检测工作台"),
                detail: model.t("Prioritize promising host stars, acquire TESS products, and audit every retained signal.", "对高价值恒星排序，获取 TESS 数据，并审计每一个保留信号。")
            )
            Spacer(minLength: 20)
            SearchField(model.t("Search TIC identifier", "搜索 TIC 编号"), text: $model.searchText)
                .controlSize(.small)
                .font(.nova(size: 12))
                .padding(.horizontal, 7)
            .frame(width: 220, height: 34)
            .background(NovaTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(NovaTheme.borderStrong, lineWidth: 1))

            Button {
                model.addSelectedToQueue(runImmediately: true)
            } label: {
                Label(model.t("Run Avril V3", "运行 Avril V3"), systemImage: "play.fill")
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(model.selectedTarget == nil)
        }
    }

    private func mission(height: CGFloat) -> some View {
        ZStack {
            MissionSceneCard(mode: $missionMode, target: model.observedTarget, active: missionMode == 0)
                .opacity(missionMode == 0 ? 1 : 0)
                .allowsHitTesting(missionMode == 0)
                .accessibilityHidden(missionMode != 0)
            LightCurveCard(payload: model.lightCurve, target: model.observedTarget, mode: $missionMode)
                .opacity(missionMode == 1 ? 1 : 0)
                .allowsHitTesting(missionMode == 1)
                .accessibilityHidden(missionMode != 1)
        }
        .frame(height: height)
        .clipped()
    }

    private var runConfiguration: some View {
        VStack(alignment: .leading, spacing: 10) {
            PanelTitle(model.t("Run Configuration", "运行配置"), subtitle: model.t("Direct NASA acquisition and local computation", "直接获取 NASA 数据并在本机计算"), symbol: "slider.horizontal.3")
            Divider().overlay(NovaTheme.border)
            if let target = model.selectedTarget {
                VStack(alignment: .leading, spacing: 3) {
                    Text(target.title)
                        .font(.nova(size: 18, weight: .semibold, design: .rounded))
                        .foregroundStyle(NovaTheme.ink)
                    Text(model.t("Queue rank #\(target.queueRank) · \(target.discoveryLane.replacingOccurrences(of: "_", with: " ").capitalized)", "队列排名 #\(target.queueRank) · \(target.discoveryLane.replacingOccurrences(of: "_", with: " "))"))
                        .font(.nova(size: 10.5))
                        .foregroundStyle(NovaTheme.inkSecondary)
                }
                HStack(spacing: 8) {
                    MetricTile(label: model.t("TESS mag", "TESS 星等"), value: target.tmag.map(\.oneDecimal) ?? "—")
                    MetricTile(label: model.t("Sectors", "观测区段"), value: "\(target.sectorCount ?? 0)", note: model.t("\(target.recentSectorCount ?? 0) recent", "\(target.recentSectorCount ?? 0) 个近期区段"))
                }
            } else {
                Text(model.t("Select a target from the ranked catalog.", "请从排名目录中选择恒星。"))
                    .font(.nova(size: 11.5))
                    .foregroundStyle(NovaTheme.inkSecondary)
            }

            HStack(spacing: 10) {
                Text(model.t("DATA SOURCE", "数据源")).font(.nova(size: 9, weight: .bold)).tracking(0.8).foregroundStyle(NovaTheme.inkTertiary)
                Spacer(minLength: 4)
                Picker("Data source", selection: $model.source) {
                    Text(model.t("Automatic", "自动")).tag("auto")
                    Text(model.t("SPOC light curves", "SPOC 光变曲线")).tag("spoc")
                    Text(model.t("FFI photometry", "FFI 测光")).tag("ffi")
                }
                .labelsHidden()
                .frame(width: 174)
            }
            Toggle(model.t("Download pixel follow-up", "下载像素复核数据"), isOn: $model.includePixelFollowup)
                .toggleStyle(.switch)
                .controlSize(.small)
                .font(.nova(size: 11))
                .tint(NovaTheme.accent)

            HStack(spacing: 8) {
                Button { model.addSelectedToQueue(runImmediately: true) } label: {
                    Label(model.t("Run Selected Target", "运行选定目标"), systemImage: "play.fill").frame(maxWidth: .infinity)
                }
                .buttonStyle(PrimaryButtonStyle(compact: true))
                .disabled(model.selectedTarget == nil)
                Button { model.addSelectedToQueue() } label: {
                    Label(model.t("Add to Task Queue", "加入任务队列"), systemImage: "text.badge.plus").frame(maxWidth: .infinity)
                }
                .buttonStyle(SecondaryButtonStyle(compact: true))
                .disabled(model.selectedTarget == nil)
            }
            Spacer(minLength: 6)
            Text(model.t("Runs execute locally. NASA archive products remain in the Avril cache and every evidence gate is auditable.", "所有计算均在本机完成。NASA 数据保留在 Avril 缓存中，每项证据门坎都可审计。"))
                .font(.nova(size: 9.5))
                .foregroundStyle(NovaTheme.inkTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .novaPanel(padding: 14)
        .clipped()
    }
}

private struct TargetCatalogView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                PanelTitle(model.t("Prioritized Target Catalog", "优先目标目录"), subtitle: model.t("Ranking prior — not a planet probability", "用于搜索排序，不是行星存在概率"), symbol: "list.number")
                Spacer()
                Picker("Lane", selection: $model.laneFilter) {
                    ForEach(model.lanes, id: \.self) { Text($0.replacingOccurrences(of: "_", with: " ").capitalized).tag($0) }
                }
                .frame(width: 170)
                .controlSize(.small)
                StatusPill(label: model.t("\(model.filteredTargets.count) targets", "\(model.filteredTargets.count) 个目标"), tint: NovaTheme.accent)
            }
            .padding(13)

            GeometryReader { tableProxy in
                ScrollView(.horizontal) {
                    VStack(spacing: 0) {
                        TargetHeaderRow()
                        ScrollView(.vertical) {
                            LazyVStack(spacing: 0) {
                                ForEach(model.filteredTargets) { target in
                                    TargetRow(target: target, selected: model.selectedTargetID == target.id)
                                        .contentShape(Rectangle())
                                        .onTapGesture { model.select(target) }
                                }
                            }
                        }
                        .scrollIndicators(.visible)
                    }
                    .frame(width: max(940, tableProxy.size.width), alignment: .leading)
                }
                .scrollIndicators(.visible)
            }
        }
        .background(NovaTheme.surface)
        .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.panelRadius).stroke(NovaTheme.border, lineWidth: 1))
        .shadow(color: NovaTheme.panelShadow, radius: 12, y: 4)
    }
}

private struct TargetHeaderRow: View {
    @EnvironmentObject private var model: AppModel
    var body: some View {
        HStack(spacing: 8) {
            header(model.t("RANK", "排名"), width: 42)
            header(model.t("TARGET", "目标"), width: 104)
            header(model.t("DISCOVERY LANE", "发现通道"), width: 144)
            header(model.t("STELLAR PROFILE", "恒星参数"), width: 160)
            header(model.t("TESS COVERAGE", "TESS 覆盖"), width: 130)
            header(model.t("MULTIPLICITY GAP", "行星数量差异"), width: 122)
            header(model.t("PRIORITY", "优先级"), width: 100)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .frame(height: 30)
        .background(NovaTheme.surfaceStrong)
        .overlay(alignment: .bottom) { Rectangle().fill(NovaTheme.border).frame(height: 1) }
    }

    private func header(_ title: String, width: CGFloat) -> some View {
        Text(title).font(.nova(size: 8.5, weight: .bold)).tracking(0.55).foregroundStyle(NovaTheme.inkTertiary).frame(width: width, alignment: .leading)
    }
}

private struct TargetRow: View {
    @EnvironmentObject private var model: AppModel
    let target: TargetRecord
    let selected: Bool

    var body: some View {
        HStack(spacing: 8) {
            Text("#\(target.queueRank)").frame(width: 42, alignment: .leading)
            Text("TIC \(target.ticID)").fontWeight(.semibold).frame(width: 104, alignment: .leading)
            Text(target.discoveryLane.replacingOccurrences(of: "_", with: " ").capitalized).frame(width: 144, alignment: .leading)
            Text("\(target.teffK.map { String(format: "%.0f K", $0) } ?? "—")  ·  \(target.radiusSolar.map { String(format: "%.2f R☉", $0) } ?? "—")")
                .frame(width: 160, alignment: .leading)
            Text(model.t("\(target.sectorCount ?? 0) sectors  ·  \(target.recentSectorCount ?? 0) recent", "\(target.sectorCount ?? 0) 个区段  ·  \(target.recentSectorCount ?? 0) 个近期"))
                .frame(width: 130, alignment: .leading)
            Text(model.t("\((target.catalogPMoreThanKnown ?? 0).percentText) more-than-known", "\((target.catalogPMoreThanKnown ?? 0).percentText) 可能多于已知"))
                .frame(width: 122, alignment: .leading)
            HStack(spacing: 7) {
                ProgressView(value: target.priority, total: max(1, target.priority > 1 ? 100 : 1))
                    .tint(NovaTheme.accent)
                    .frame(width: 58)
                Text(target.priority <= 1 ? target.priority.percentText : target.priority.oneDecimal)
                    .monospacedDigit()
                    .frame(width: 34, alignment: .trailing)
            }
            .frame(width: 100, alignment: .leading)
            Spacer(minLength: 0)
            Button {
                model.select(target)
                model.addSelectedToQueue()
            } label: {
                Image(systemName: "plus").font(.nova(size: 10, weight: .semibold))
            }
            .buttonStyle(SecondaryButtonStyle(compact: true))
        }
        .font(.nova(size: 10.5))
        .foregroundStyle(NovaTheme.inkSecondary)
        .padding(.horizontal, 12)
        .frame(height: 42)
        .background(selected ? NovaTheme.accentSoft : NovaTheme.surface)
        .overlay(alignment: .leading) {
            if selected { Rectangle().fill(NovaTheme.accent).frame(width: 3) }
        }
        .overlay(alignment: .bottom) { Rectangle().fill(NovaTheme.border.opacity(0.65)).frame(height: 1) }
    }
}

private struct TaskQueueCard: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 11) {
            HStack {
                PanelTitle(model.t("Task Queue", "任务队列"), subtitle: model.t("Local sequential execution", "本地顺序执行"), symbol: "list.bullet.rectangle")
                Spacer()
                StatusPill(label: model.t("\(model.tasks.filter { $0.state == .queued }.count) waiting", "\(model.tasks.filter { $0.state == .queued }.count) 个等待"), tint: NovaTheme.violet)
            }
            if model.tasks.isEmpty {
                EmptyScienceState(symbol: "tray", title: model.t("Queue is empty", "队列为空"), message: model.t("Add a recommended target or run the current selection.", "加入推荐目标，或直接运行当前选择。"))
                    .frame(height: 108)
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(model.tasks.reversed()) { task in
                            VStack(alignment: .leading, spacing: 7) {
                                HStack {
                                    Circle().fill(task.state.tint).frame(width: 7, height: 7)
                                    Text(task.target.title).font(.nova(size: 11, weight: .semibold)).foregroundStyle(NovaTheme.ink)
                                    Spacer()
                                    Text(task.state.rawValue.uppercased()).font(.nova(size: 8.5, weight: .bold)).foregroundStyle(task.state.tint)
                                    if [.queued, .running].contains(task.state) {
                                        Button { model.stop(task: task) } label: { Image(systemName: "stop.fill") }
                                            .buttonStyle(.plain).foregroundStyle(NovaTheme.red)
                                    }
                                }
                                ProgressView(value: task.progress).tint(task.state.tint)
                                Text(task.message).font(.nova(size: 9.5)).foregroundStyle(NovaTheme.inkTertiary).lineLimit(2)
                            }
                            .padding(10)
                            .background(NovaTheme.surfaceMuted)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(NovaTheme.border, lineWidth: 1))
                            .contentShape(Rectangle())
                            .onTapGesture {
                                guard task.state == .complete else { return }
                                model.select(task.target)
                                model.openRunRecord(jobID: task.jobID)
                            }
                            .help(task.state == .complete
                                ? model.t("Open the completed evidence record", "打开已完成的证据记录")
                                : "")
                        }
                    }
                }
                .frame(maxHeight: 205)
            }
        }
        .novaPanel(padding: 13)
    }
}

private struct PipelineCard: View {
    @EnvironmentObject private var model: AppModel

    var active: ScienceTask? { model.tasks.first { $0.state == .running } }
    var activeStage: String { active?.stage.lowercased() ?? "" }

    var body: some View {
        VStack(spacing: 10) {
            PanelTitle(model.t("Evidence Pipeline", "证据流程"), subtitle: model.t("Non-destructive, auditable stages", "非破坏性、可审计流程"), symbol: "point.3.connected.trianglepath.dotted")
            WorkflowStep(index: 1, title: model.t("Acquire", "获取"), detail: model.t("NASA archive products", "NASA 归档数据"), state: state(for: ["discover", "download", "fetch", "starting"]))
            WorkflowStep(index: 2, title: model.t("Condition", "预处理"), detail: model.t("Multi-branch detrending", "多分支去趋势"), state: state(for: ["prepare", "condition", "detrend"]))
            WorkflowStep(index: 3, title: model.t("Search", "搜索"), detail: model.t("BLS, TLS, event channel", "BLS、TLS 与独立事件"), state: state(for: ["search", "period"]))
            WorkflowStep(index: 4, title: model.t("Vet", "排假"), detail: model.t("Gap, red-noise, pixel gates", "缺口、红噪声与像素门坎"), state: state(for: ["vet", "pixel", "evidence"]))
            WorkflowStep(index: 5, title: model.t("Audit", "审计"), detail: model.t("Candidate ledger and report", "候选记录与报告"), state: active == nil && model.tasks.contains { $0.state == .complete } ? .done : state(for: ["complete", "audit"]))
        }
        .novaPanel(padding: 13)
    }

    private func state(for tokens: [String]) -> WorkflowStep.StepState {
        guard let active else { return .waiting }
        if tokens.contains(where: { activeStage.contains($0) }) { return .active }
        if active.progress > 0.82 && !tokens.contains("complete") { return .done }
        return .waiting
    }
}
