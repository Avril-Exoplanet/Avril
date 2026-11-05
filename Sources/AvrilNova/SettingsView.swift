import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeading(
                    eyebrow: model.t("Application preferences", "应用偏好设置"),
                    title: model.t("Settings", "设置"),
                    detail: model.t("Configure presentation and local execution without changing the frozen science gates.", "配置界面和本地执行，不会修改已冻结的科学门坎。")
                )
                settingPanel(model.t("Interface", "界面"), symbol: "macwindow") {
                    settingRow(model.t("Language", "语言"), detail: model.t("Application labels and explanations", "应用标签与说明文字")) {
                        Picker("Language", selection: $model.language) {
                            Text("English").tag("English")
                            Text("简体中文").tag("简体中文")
                        }
                        .labelsHidden().frame(width: 150)
                    }
                    settingRow(model.t("Target inspector", "目标检查器"), detail: model.t("Show the contextual property panel", "显示上下文属性面板")) {
                        Toggle("", isOn: $model.showInspector).labelsHidden().toggleStyle(.switch).tint(NovaTheme.accent)
                    }
                }
                settingPanel(model.t("Science Runtime", "科学运行时"), symbol: "cpu") {
                    settingRow(model.t("Execution", "执行状态"), detail: model.t("Avril V3 runs only on this Mac", "Avril V3 仅在这台 Mac 上运行")) {
                        StatusPill(label: model.engineState.label, tint: { if case .ready = model.engineState { NovaTheme.green } else { NovaTheme.amber } }())
                    }
                    settingRow(model.t("Pixel follow-up", "像素复核"), detail: model.t("Acquire spatial evidence for retained candidates", "为保留候选获取空间证据")) {
                        Toggle("", isOn: $model.includePixelFollowup).labelsHidden().toggleStyle(.switch).tint(NovaTheme.accent)
                    }
                    settingRow(model.t("Frozen gates", "冻结门坎"), detail: model.t("Gap-edge, red-noise, harmonic, odd-even, and centroid rules", "缺口边界、红噪声、谐波、奇偶与质心规则")) {
                        Text("Avril V3").font(.nova(size: 11, weight: .semibold)).foregroundStyle(NovaTheme.green)
                    }
                }
                settingPanel(model.t("Data", "数据"), symbol: "internaldrive") {
                    Text(model.t("Science products are stored in the local Application Support directory. Removing the app does not silently delete downloaded NASA products or audit records.", "科学数据保存在本地 Application Support 目录中。删除应用不会悄然删除已下载的 NASA 数据或审计记录。"))
                        .font(.nova(size: 11))
                        .foregroundStyle(NovaTheme.inkSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: 900, alignment: .leading)
            .padding(22)
        }
        .background(NovaTheme.canvas)
    }

    private func settingPanel<Content: View>(_ title: String, symbol: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            PanelTitle(title, symbol: symbol)
            Divider().overlay(NovaTheme.border)
            content()
        }
        .novaPanel()
    }

    private func settingRow<Trailing: View>(_ title: String, detail: String, @ViewBuilder trailing: () -> Trailing) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.nova(size: 12, weight: .medium)).foregroundStyle(NovaTheme.ink)
                Text(detail).font(.nova(size: 10.5)).foregroundStyle(NovaTheme.inkTertiary)
            }
            Spacer()
            trailing()
        }
        .frame(minHeight: 42)
    }
}
