import SwiftUI

struct CandidateReviewView: View {
    @EnvironmentObject private var model: AppModel

    private var completed: [ScienceTask] {
        model.tasks.filter { $0.state == .complete }
    }

    private var summaries: [(task: ScienceTask, evidence: EvidenceSummary)] {
        completed.map { ($0, EvidenceSummary(task: $0)) }
    }

    private var totalAuditRows: Int {
        summaries.compactMap(\.evidence.auditRows).reduce(0, +)
    }

    private var totalDiagnosticRows: Int {
        summaries.compactMap(\.evidence.diagnosticRows).reduce(0, +)
    }

    var body: some View {
        VStack(spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                SectionHeading(
                    eyebrow: model.t("Evidence ledger", "证据记录"),
                    title: model.t("Evidence Review", "证据复核"),
                    detail: model.t(
                        "Review the disposition of every tested ephemeris. An audit row is not a planet; an unexplained row is still unverified.",
                        "复核每一条已测试星历的处置结果。审计记录不等于行星；暂无解释的记录仍未得到验证。"
                    )
                )
                Spacer(minLength: 12)
                if !completed.isEmpty {
                    HStack(spacing: 8) {
                        StatusPill(
                            label: model.t("\(totalAuditRows) audit rows", "\(totalAuditRows) 条审计记录"),
                            tint: NovaTheme.accent
                        )
                        if totalDiagnosticRows > 0 {
                            StatusPill(
                                label: model.t("\(totalDiagnosticRows) sent to diagnostics", "\(totalDiagnosticRows) 条进入诊断"),
                                tint: NovaTheme.cyan
                            )
                        }
                    }
                }
            }

            VStack(spacing: 0) {
                headerRow

                if completed.isEmpty {
                    EmptyScienceState(
                        symbol: "waveform.path.ecg.rectangle",
                        title: model.t("No completed evidence reports", "暂无已完成的证据报告"),
                        message: model.t(
                            "Completed runs appear here with their audit-ledger size and evidence disposition.",
                            "已完成的任务将在这里显示审计记录规模和证据处置结果。"
                        )
                    )
                } else {
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(summaries, id: \.task.id) { entry in
                                evidenceRow(task: entry.task, evidence: entry.evidence)
                            }
                        }
                    }
                }
            }
            .background(NovaTheme.surface)
            .clipShape(RoundedRectangle(cornerRadius: NovaTheme.panelRadius))
            .overlay(RoundedRectangle(cornerRadius: NovaTheme.panelRadius).stroke(NovaTheme.border))
        }
        .padding(18)
        .background(NovaTheme.canvas)
    }

    private var headerRow: some View {
        HStack(spacing: 12) {
            Text(model.t("TARGET", "目标")).frame(width: 142, alignment: .leading)
            Text(model.t("RUN", "任务")).frame(width: 82, alignment: .leading)
            Text(model.t("AUDIT ROWS", "审计记录")).frame(width: 92, alignment: .leading)
            Text(model.t("DIAGNOSTIC SET", "诊断集")).frame(width: 112, alignment: .leading)
            Text(model.t("EVIDENCE DISPOSITION", "证据处置")).frame(maxWidth: .infinity, alignment: .leading)
            Text(model.t("RUNTIME", "运行时间")).frame(width: 92, alignment: .leading)
        }
        .font(.nova(size: 8.5, weight: .bold))
        .tracking(0.65)
        .foregroundStyle(NovaTheme.inkTertiary)
        .padding(.horizontal, 14)
        .frame(height: 34)
        .background(NovaTheme.surfaceStrong)
    }

    private func evidenceRow(task: ScienceTask, evidence: EvidenceSummary) -> some View {
        HStack(spacing: 12) {
            Text(task.target.title)
                .fontWeight(.semibold)
                .foregroundStyle(NovaTheme.ink)
                .frame(width: 142, alignment: .leading)

            StatusPill(
                label: model.t("Run complete", "任务完成"),
                tint: NovaTheme.green
            )
            .frame(width: 82, alignment: .leading)

            metricText(
                evidence.auditRows.map { model.t("\($0) tested", "\($0) 条已测试") } ?? "—"
            )
            .frame(width: 92, alignment: .leading)

            metricText(diagnosticLabel(evidence))
                .frame(width: 112, alignment: .leading)

            VStack(alignment: .leading, spacing: 3) {
                Text(dispositionLabel(evidence))
                    .foregroundStyle(NovaTheme.inkSecondary)
                    .lineLimit(1)
                Text(diagnosticStatus(evidence))
                    .font(.nova(size: 9.5, weight: .medium))
                    .foregroundStyle(NovaTheme.inkTertiary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            metricText(task.runtimeSeconds.map { String(format: "%.1f s", $0) } ?? "—")
                .frame(width: 92, alignment: .leading)
        }
        .font(.nova(size: 11))
        .padding(.horizontal, 14)
        .frame(minHeight: 54)
        .overlay(alignment: .bottom) { Rectangle().fill(NovaTheme.border).frame(height: 1) }
    }

    private func metricText(_ value: String) -> some View {
        Text(value)
            .font(.nova(size: 10.5, weight: .medium, design: .monospaced))
            .foregroundStyle(NovaTheme.inkSecondary)
            .lineLimit(1)
    }

    private func diagnosticLabel(_ evidence: EvidenceSummary) -> String {
        guard let count = evidence.diagnosticRows else { return "—" }
        return model.t("\(count) unverified", "\(count) 条未验证")
    }

    private func dispositionLabel(_ evidence: EvidenceSummary) -> String {
        var parts: [String] = []
        if let gap = evidence.gapDominated {
            parts.append(model.t("\(gap) gap-dominated", "\(gap) 条缺测主导"))
        }
        if let incoherent = evidence.eventIncoherent {
            parts.append(model.t("\(incoherent) event-incoherent", "\(incoherent) 条事件不一致"))
        }
        return parts.isEmpty
            ? model.t("Disposition breakdown unavailable", "暂无处置分类明细")
            : parts.joined(separator: "  ·  ")
    }

    private func diagnosticStatus(_ evidence: EvidenceSummary) -> String {
        if let complete = evidence.diagnosticsComplete,
           let total = evidence.diagnosticsTotal {
            return model.t("\(complete)/\(total) diagnostic rows processed", "已处理 \(complete)/\(total) 条诊断记录")
        }
        if let rows = evidence.diagnosticRows {
            return model.t("\(rows) rows advanced; no confirmation implied", "\(rows) 条进入诊断，不代表已确认")
        }
        return model.t("Audit ledger sealed", "审计记录已封存")
    }
}

private struct EvidenceSummary: Hashable {
    let auditRows: Int?
    let diagnosticRows: Int?
    let gapDominated: Int?
    let eventIncoherent: Int?
    let diagnosticsComplete: Int?
    let diagnosticsTotal: Int?

    init(task: ScienceTask) {
        let funnel = task.events.reversed().compactMap { $0.metrics?.meaningfulCandidateFunnel }.first
        auditRows = funnel?.inputCandidates
            ?? task.events.reversed().compactMap { $0.metrics?.allCandidates }.first
            ?? task.candidateCount
        diagnosticRows = funnel?.meaningfulCandidates
        gapDominated = funnel?.dispositionCounts?.gapDominated
        eventIncoherent = funnel?.dispositionCounts?.eventIncoherent

        let vettingEvents = task.events.reversed().filter { event in
            event.stage?.lowercased() == "vetting"
        }
        diagnosticsComplete = vettingEvents.compactMap { $0.metrics?.candidatesCompleted }.first
        diagnosticsTotal = vettingEvents.compactMap { $0.metrics?.candidatesTotal }.first
    }
}
