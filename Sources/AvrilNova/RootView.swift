import AppKit
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            TopBar()
            CommandBar()
            HStack(spacing: 0) {
                NavigationRail()
                    .frame(width: 92)
                PanelDivider().frame(width: 1)
                content
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .layoutPriority(1)
                if model.showInspector && model.section != .settings && model.section != .review {
                    PanelDivider().frame(width: 1)
                    TargetInspector()
                        .frame(width: 284)
                }
            }
            PanelDivider().frame(height: 1)
            StatusBar()
        }
        .background(NovaTheme.canvas)
        .foregroundStyle(NovaTheme.ink)
        .alert("Avril Nova", isPresented: Binding(
            get: { model.errorBanner != nil },
            set: { if !$0 { model.errorBanner = nil } }
        )) {
            Button("Dismiss", role: .cancel) { model.errorBanner = nil }
        } message: {
            Text(model.errorBanner ?? "")
        }
    }

    @ViewBuilder
    private var content: some View {
        switch model.section {
        case .workbench: WorkbenchView()
        case .pixels: PixelValidationView()
        case .review: RunRecordsView()
        case .settings: SettingsView()
        }
    }
}

private struct TopBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 10) {
                ZStack {
                    Circle().stroke(NovaTheme.accent.opacity(0.35), lineWidth: 1)
                    Circle().trim(from: 0.08, to: 0.72).stroke(NovaTheme.accent, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    Circle().fill(NovaTheme.cyan).frame(width: 5, height: 5).offset(x: 10, y: -6)
                }
                .frame(width: 28, height: 28)
                VStack(alignment: .leading, spacing: 0) {
                    Text("AVRIL NOVA")
                        .font(.nova(size: 12, weight: .bold))
                        .tracking(1.05)
                        .foregroundStyle(NovaTheme.ink)
                    Text("EXOPLANET WORKSTATION")
                        .font(.nova(size: 7.5, weight: .semibold))
                        .tracking(0.9)
                        .foregroundStyle(NovaTheme.inkTertiary)
                }
            }
            .padding(.leading, 76)
            Spacer()
            Button {
                if case .failed = model.engineState { model.retryEngine() }
            } label: {
                HStack(spacing: 6) {
                    Circle().fill(engineTint).frame(width: 6, height: 6)
                    Text(localizedEngineLabel)
                        .font(.nova(size: 10.5, weight: .medium))
                }
                .foregroundStyle(NovaTheme.inkSecondary)
                .padding(.horizontal, 9)
                .frame(height: 27)
                .background(NovaTheme.surface.opacity(0.82))
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(NovaTheme.border, lineWidth: 1))
            }
            .buttonStyle(.plain)
            .help(model.t("Click to retry when unavailable", "引擎不可用时点击重试"))
            Button {
                model.showInspector.toggle()
            } label: {
                Image(systemName: "sidebar.right")
                    .font(.nova(size: 13, weight: .medium))
                    .frame(width: 30, height: 30)
            }
            .buttonStyle(SecondaryButtonStyle(compact: true))
            .help("Toggle Inspector")
        }
        .frame(height: 50)
        .padding(.trailing, 12)
        .background {
            EffectView(.headerView, blendingMode: .withinWindow)
                .overlay(NovaTheme.chrome.opacity(0.74))
        }
    }

    private var engineTint: Color {
        switch model.engineState {
        case .starting: NovaTheme.amber
        case .ready: NovaTheme.green
        case .failed: NovaTheme.red
        }
    }

    private var localizedEngineLabel: String {
        switch model.engineState {
        case .starting: model.t("Starting science engine", "正在启动科学引擎")
        case .ready(let version): model.t("Avril \(version) ready", "Avril \(version) 已就绪")
        case .failed: model.t("Engine unavailable · Retry", "引擎不可用 · 重试")
        }
    }
}

private struct CommandBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: 92)
            toolGroup {
                command("folder", model.t("Open FITS", "打开 FITS")) {}
                command("square.and.arrow.down", model.t("Import Queue", "导入队列")) {}
            }
            divider
            toolGroup {
                command("arrow.clockwise", model.t("Refresh Catalog", "刷新目录")) { Task { await model.loadTargets() } }
                command("plus.rectangle.on.folder", model.t("Queue Target", "加入任务")) { model.addSelectedToQueue() }
            }
            divider
            toolGroup {
                command("play.fill", model.t("Run Avril V3", "运行 Avril V3"), emphasized: true) { model.addSelectedToQueue(runImmediately: true) }
                command("stop.fill", model.t("Stop", "停止")) {
                    if let running = model.tasks.first(where: { $0.state == .running }) { model.stop(task: running) }
                }
            }
            Spacer()
            Text(model.t("LOCAL SCIENCE SESSION", "本地科学会话"))
                .font(.nova(size: 8.5, weight: .semibold))
                .tracking(0.9)
                .foregroundStyle(NovaTheme.inkTertiary)
                .padding(.trailing, 13)
        }
        .frame(height: 38)
        .background {
            EffectView(.headerView, blendingMode: .withinWindow)
                .overlay(NovaTheme.chromeRaised.opacity(0.76))
        }
        .overlay(alignment: .bottom) { PanelDivider().frame(height: 1) }
    }

    private var divider: some View {
        Rectangle().fill(NovaTheme.chromeBorder).frame(width: 1, height: 24).padding(.horizontal, 6)
    }

    private func toolGroup<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        HStack(spacing: 3) { content() }.padding(.horizontal, 4)
    }

    private func command(_ symbol: String, _ title: String, emphasized: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: symbol).font(.nova(size: 10.5, weight: .semibold))
                Text(title).font(.nova(size: 10.5, weight: .medium))
            }
            .foregroundStyle(emphasized ? .white : NovaTheme.inkSecondary)
            .padding(.horizontal, 9)
            .frame(height: 28)
            .background(emphasized ? NovaTheme.accent : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
        .buttonStyle(.plain)
    }
}

private struct NavigationRail: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 5) {
            ForEach(WorkspaceSection.allCases) { section in
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) { model.section = section }
                } label: {
                    VStack(spacing: 5) {
                        Image(systemName: section.symbol)
                            .font(.nova(size: 16, weight: model.section == section ? .semibold : .regular))
                        Text(localizedTitle(section))
                            .font(.nova(size: 9.5, weight: model.section == section ? .semibold : .medium))
                            .lineLimit(1)
                    }
                    .foregroundStyle(model.section == section ? NovaTheme.accentDark : NovaTheme.inkTertiary)
                    .frame(width: 74, height: 58)
                    .background(model.section == section ? NovaTheme.accentSoft : .clear)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .overlay(alignment: .leading) {
                        if model.section == section {
                            Capsule().fill(NovaTheme.accent).frame(width: 3, height: 27).offset(x: -6)
                        }
                    }
                }
                .buttonStyle(.plain)
            }
            Spacer()
            Text("v0.2")
                .font(.nova(size: 8.5, weight: .medium))
                .foregroundStyle(NovaTheme.inkTertiary)
                .padding(.bottom, 10)
        }
        .padding(.horizontal, 8)
        .padding(.top, 10)
        .frame(width: 92)
        .background(NovaTheme.sidebar)
    }

    private func localizedTitle(_ section: WorkspaceSection) -> String {
        switch section {
        case .workbench: model.t("Workbench", "工作台")
        case .pixels: model.t("Pixels", "像素")
        case .review: model.t("Records", "记录")
        case .settings: model.t("Settings", "设置")
        }
    }
}

private struct TargetInspector: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(model.t("TARGET INSPECTOR", "目标检查器"))
                    .font(.nova(size: 9.5, weight: .bold))
                    .tracking(1)
                    .foregroundStyle(NovaTheme.inkTertiary)
                Spacer()
                Button { model.showInspector = false } label: {
                    Image(systemName: "xmark").font(.nova(size: 10, weight: .semibold))
                }
                .buttonStyle(.plain)
                .foregroundStyle(NovaTheme.inkTertiary)
            }
            .padding(13)
            .background(NovaTheme.surfaceStrong)
            Rectangle().fill(NovaTheme.border).frame(height: 1)

            if let target = model.selectedTarget {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(target.title)
                                .font(.nova(size: 19, weight: .semibold, design: .rounded))
                            Text("Frozen discovery queue rank #\(target.queueRank)")
                                .font(.nova(size: 10.5))
                                .foregroundStyle(NovaTheme.inkTertiary)
                        }
                        StatusPill(label: target.discoveryLane.replacingOccurrences(of: "_", with: " ").capitalized, tint: NovaTheme.cyan)
                        inspectorSection("STELLAR PROFILE") {
                            property("Effective temperature", target.teffK.map { String(format: "%.0f K", $0) } ?? "—")
                            property("Stellar radius", target.radiusSolar.map { String(format: "%.3f R☉", $0) } ?? "—")
                            property("Stellar mass", target.massSolar.map { String(format: "%.3f M☉", $0) } ?? "—")
                            property("TESS magnitude", target.tmag.map(\.oneDecimal) ?? "—")
                        }
                        inspectorSection("DISCOVERY PRIOR") {
                            property("Known planets", "\(target.knownPlanets ?? 0)")
                            property("Expected lower bound", target.catalogExpectedCountLowerBound.map { String(format: "%.2f", $0) } ?? "—")
                            property("P(more than known)", (target.catalogPMoreThanKnown ?? 0).percentText)
                            property("Detectability percentile", (target.detectabilityPercentile ?? 0).percentText)
                            property("Model applicability", target.modelApplicability ?? "—")
                        }
                        inspectorSection("TESS COVERAGE") {
                            property("Observed sectors", "\(target.sectorCount ?? 0)")
                            property("Recent sectors", "\(target.recentSectorCount ?? 0)")
                            property("Sector span", "\(target.sectorSpan ?? 0)")
                            property("Multi-epoch", target.multiEpochCoverage == true ? "Yes" : "No")
                        }
                        Button { model.addSelectedToQueue(runImmediately: true) } label: {
                            Label("Run Avril V3", systemImage: "play.fill").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(PrimaryButtonStyle())
                    }
                    .padding(14)
                }
            } else {
                EmptyScienceState(symbol: "scope", title: "No target selected", message: "Choose a ranked star in the workbench.")
            }
        }
        .background(NovaTheme.surfaceMuted)
    }

    private func inspectorSection<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.nova(size: 8.5, weight: .bold)).tracking(0.8).foregroundStyle(NovaTheme.inkTertiary)
            VStack(spacing: 0) { content() }
                .background(NovaTheme.surface)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(NovaTheme.border, lineWidth: 1))
        }
    }

    private func property(_ key: String, _ value: String) -> some View {
        HStack {
            Text(key).foregroundStyle(NovaTheme.inkSecondary)
            Spacer()
            Text(value).fontWeight(.medium).foregroundStyle(NovaTheme.ink).monospacedDigit()
        }
        .font(.nova(size: 10.5))
        .padding(.horizontal, 10)
        .frame(height: 29)
        .overlay(alignment: .bottom) { Rectangle().fill(NovaTheme.border.opacity(0.65)).frame(height: 1) }
    }
}

private struct StatusBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill({ if case .ready = model.engineState { NovaTheme.green } else { NovaTheme.amber } }())
                .frame(width: 6, height: 6)
            Text({
                switch model.engineState {
                case .starting: model.t("Starting engine", "正在启动引擎")
                case .ready(let version): model.t("Avril \(version) ready", "Avril \(version) 已就绪")
                case .failed: model.t("Engine unavailable", "引擎不可用")
                }
            }())
            Text("·")
            Text(model.t("\(model.targets.count) ranked targets", "\(model.targets.count) 个排名目标"))
            Text("·")
            Text(model.t("\(model.tasks.filter { $0.state == .running }.count) active", "\(model.tasks.filter { $0.state == .running }.count) 个运行中"))
            Spacer()
            Text(model.t("LOCAL COMPUTE · AUDITABLE PIPELINE", "本地计算 · 可审计流程"))
                .fontWeight(.semibold)
                .tracking(0.65)
        }
        .font(.nova(size: 9.5))
        .foregroundStyle(NovaTheme.inkTertiary)
        .padding(.horizontal, 12)
        .frame(height: 25)
        .background(NovaTheme.surfaceStrong)
    }
}

struct WindowConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { configure(view.window) }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async { configure(nsView.window) }
    }

    private func configure(_ window: NSWindow?) {
        guard let window else { return }
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.styleMask.insert(.fullSizeContentView)
        window.isMovableByWindowBackground = false
        window.minSize = NSSize(width: 1280, height: 760)
        window.backgroundColor = NSColor(NovaTheme.canvas)
        window.tabbingMode = .disallowed
    }
}
