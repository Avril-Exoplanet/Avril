import AppKit
import Combine
import Foundation

@MainActor
final class AppModel: ObservableObject {
    @Published var section: WorkspaceSection = .workbench
    @Published var engineState: EngineState = .starting
    @Published var targets: [TargetRecord] = []
    @Published var selectedTargetID: Int?
    @Published var searchText = ""
    @Published var laneFilter = "All lanes"
    @Published var source = "auto"
    @Published var includePixelFollowup = false
    @Published var tasks: [ScienceTask] = []
    @Published var runRecords: [RunRecordSummary] = []
    @Published var selectedRunID: String?
    @Published var selectedCandidateID: String?
    @Published var selectedScienceResult: ScienceResultDocument?
    @Published var selectedExecutionLog: ExecutionLogDocument?
    @Published var selectedScienceTimeline: ScienceTimelineDocument?
    @Published var selectedRunLightCurve: LightCurvePayload?
    @Published var selectedCandidateDetail: ResultCandidate?
    @Published var selectedCandidateFolds: CandidateFoldDocument?
    @Published var isLoadingRunRecords = false
    @Published var isLoadingScienceResult = false
    @Published var isLoadingExecutionLog = false
    @Published var isLoadingScienceTimeline = false
    @Published var isLoadingCandidateDetail = false
    @Published var isLoadingCandidateFolds = false
    @Published var isExportingRunArtifact = false
    @Published var scienceResultError: String?
    @Published var executionLogError: String?
    @Published var scienceTimelineError: String?
    @Published var candidateFoldError: String?
    @Published var lightCurve: LightCurvePayload?
    @Published private(set) var lightCurveTaskID: UUID?
    @Published var isLoadingTargets = false
    @Published var errorBanner: String?
    @Published var language: String {
        didSet { UserDefaults.standard.set(language, forKey: "AvrilNova.language") }
    }
    @Published var showInspector = true

    private let engine = EngineController()
    private var runner: Task<Void, Never>?
    private var healthMonitor: Task<Void, Never>?
    private var isRecoveringEngine = false
    private var consecutiveHealthFailures = 0
    private var scienceResultRequestID: UUID?
    private var executionLogRequestID: UUID?
    private var scienceTimelineRequestID: UUID?
    private var candidateDetailRequestID: UUID?
    private var candidateFoldRequestID: UUID?

    var selectedTarget: TargetRecord? {
        guard let selectedTargetID else { return nil }
        return targets.first { $0.id == selectedTargetID }
    }

    var activeTask: ScienceTask? {
        tasks.first { $0.state == .running }
    }

    var focusedRunTask: ScienceTask? {
        activeTask ?? tasks.last
    }

    var observedTarget: TargetRecord? {
        if let lightCurveTaskID,
           let task = tasks.first(where: { $0.id == lightCurveTaskID }) {
            return task.target
        }
        return activeTask?.target ?? selectedTarget
    }

    var lanes: [String] {
        ["All lanes"] + Array(Set(targets.map(\.discoveryLane))).sorted()
    }

    var filteredTargets: [TargetRecord] {
        targets.filter { target in
            let laneMatches = laneFilter == "All lanes" || target.discoveryLane == laneFilter
            let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "TIC", with: "", options: .caseInsensitive)
                .trimmingCharacters(in: .whitespaces)
            let searchMatches = query.isEmpty || String(target.ticID).contains(query)
            return laneMatches && searchMatches
        }
    }

    init() {
        language = UserDefaults.standard.string(forKey: "AvrilNova.language") ?? "English"
        Task { [weak self] in
            guard let self else { return }
            await self.bootstrap()
            self.startHealthMonitor()
        }
    }

    var isChinese: Bool { language == "简体中文" }

    func t(_ english: String, _ chinese: String) -> String {
        isChinese ? chinese : english
    }

    func bootstrap() async {
        guard !isRecoveringEngine else { return }
        isRecoveringEngine = true
        defer { isRecoveringEngine = false }
        engineState = .starting
        do {
            let version = try await engine.start()
            consecutiveHealthFailures = 0
            engineState = .ready(version: version)
            await loadTargets()
            // Restore the lightweight history index during launch.  The
            // multi-megabyte scientific report is decoded only when the user
            // opens Run Records, keeping the workbench responsive at startup.
            await loadRunRecords(loadSelection: false)
        } catch {
            engineState = .failed(error.localizedDescription)
            errorBanner = error.localizedDescription
        }
    }

    func retryEngine() {
        Task { await recoverEngine(force: true) }
    }

    func loadTargets() async {
        guard !isLoadingTargets else { return }
        isLoadingTargets = true
        defer { isLoadingTargets = false }
        do {
            var rows: [TargetRecord] = []
            var offset = 0
            var total = Int.max
            while offset < total {
                let page = try await engine.targets(offset: offset)
                total = page.total
                rows.append(contentsOf: page.items)
                offset += page.items.count
                if page.items.isEmpty { break }
            }
            targets = rows
            if selectedTargetID == nil { selectedTargetID = rows.first?.id }
        } catch {
            errorBanner = "Target catalog: \(error.localizedDescription)"
        }
    }

    func select(_ target: TargetRecord) {
        selectedTargetID = target.id
    }

    func loadRunRecords(loadSelection: Bool = false) async {
        guard !isLoadingRunRecords else { return }
        isLoadingRunRecords = true
        defer { isLoadingRunRecords = false }
        do {
            let page = try await engine.runRecords()
            runRecords = page.items
            if selectedRunID == nil || !page.items.contains(where: { $0.id == selectedRunID }) {
                selectedRunID = page.items.first(where: \.isComplete)?.id ?? page.items.first?.id
            }
            if loadSelection, let selectedRunID {
                async let result: Void = loadScienceResult(runID: selectedRunID)
                async let log: Void = loadExecutionLog(runID: selectedRunID)
                async let timeline: Void = loadScienceTimeline(runID: selectedRunID)
                _ = await (result, log, timeline)
            }
        } catch {
            scienceResultError = error.localizedDescription
        }
    }

    func selectRun(_ record: RunRecordSummary) {
        selectedRunID = record.id
        selectedCandidateID = nil
        selectedScienceResult = nil
        selectedExecutionLog = nil
        selectedScienceTimeline = nil
        selectedRunLightCurve = nil
        selectedCandidateDetail = nil
        selectedCandidateFolds = nil
        candidateFoldError = nil
        scienceResultError = nil
        executionLogError = nil
        scienceTimelineError = nil
        Task {
            async let log: Void = loadExecutionLog(runID: record.id)
            if record.isComplete {
                async let result: Void = loadScienceResult(runID: record.id)
                async let timeline: Void = loadScienceTimeline(runID: record.id)
                _ = await (log, result, timeline)
            } else {
                await log
            }
        }
    }

    func openRunRecord(jobID: String?) {
        section = .review
        if let jobID, let record = runRecords.first(where: { $0.id == jobID }) {
            selectRun(record)
        } else {
            Task { await loadRunRecords(loadSelection: true) }
        }
    }

    func loadScienceResult(runID: String) async {
        guard runRecords.first(where: { $0.id == runID })?.isComplete == true else { return }
        let requestID = UUID()
        scienceResultRequestID = requestID
        isLoadingScienceResult = true
        scienceResultError = nil
        defer {
            if scienceResultRequestID == requestID { isLoadingScienceResult = false }
        }
        do {
            let result = try await engine.scienceResult(jobID: runID)
            guard selectedRunID == runID, scienceResultRequestID == requestID else { return }
            selectedScienceResult = result
            if selectedCandidateID == nil {
                selectedCandidateID = result.auditCandidates.first?.id
            }
            if let candidate = result.auditCandidates.first(where: { $0.id == selectedCandidateID }) {
                async let detail: Void = loadCandidateDetail(runID: runID, candidateID: candidate.candidateID)
                async let folds: Void = loadCandidateFolds(runID: runID, candidateID: candidate.candidateID)
                _ = await (detail, folds)
            }
            let runLightCurve = try? await engine.lightCurve(jobID: runID)
            guard selectedRunID == runID, scienceResultRequestID == requestID else { return }
            selectedRunLightCurve = runLightCurve
        } catch {
            guard selectedRunID == runID, scienceResultRequestID == requestID else { return }
            scienceResultError = error.localizedDescription
        }
    }

    func loadExecutionLog(runID: String) async {
        let requestID = UUID()
        executionLogRequestID = requestID
        isLoadingExecutionLog = true
        executionLogError = nil
        defer {
            if executionLogRequestID == requestID { isLoadingExecutionLog = false }
        }
        do {
            let log = try await engine.executionLog(jobID: runID)
            guard selectedRunID == runID, executionLogRequestID == requestID else { return }
            selectedExecutionLog = log
        } catch {
            guard selectedRunID == runID, executionLogRequestID == requestID else { return }
            executionLogError = error.localizedDescription
        }
    }

    func loadScienceTimeline(runID: String) async {
        guard runRecords.first(where: { $0.id == runID })?.isComplete == true else { return }
        let requestID = UUID()
        scienceTimelineRequestID = requestID
        isLoadingScienceTimeline = true
        scienceTimelineError = nil
        defer {
            if scienceTimelineRequestID == requestID { isLoadingScienceTimeline = false }
        }
        do {
            let timeline = try await engine.scienceTimeline(jobID: runID)
            guard selectedRunID == runID, scienceTimelineRequestID == requestID else { return }
            selectedScienceTimeline = timeline
        } catch {
            guard selectedRunID == runID, scienceTimelineRequestID == requestID else { return }
            scienceTimelineError = error.localizedDescription
        }
    }

    func selectCandidate(_ candidate: ResultCandidate) {
        selectedCandidateID = candidate.id
        selectedCandidateDetail = nil
        selectedCandidateFolds = nil
        candidateFoldError = nil
        guard let runID = selectedRunID else { return }
        Task {
            async let detail: Void = loadCandidateDetail(runID: runID, candidateID: candidate.candidateID)
            async let folds: Void = loadCandidateFolds(runID: runID, candidateID: candidate.candidateID)
            _ = await (detail, folds)
        }
    }

    func loadCandidateDetail(runID: String, candidateID: String) async {
        let requestID = UUID()
        candidateDetailRequestID = requestID
        isLoadingCandidateDetail = true
        defer {
            if candidateDetailRequestID == requestID { isLoadingCandidateDetail = false }
        }
        do {
            let candidate = try await engine.candidateDetail(jobID: runID, candidateID: candidateID)
            guard selectedRunID == runID,
                  selectedScienceResult?.auditCandidates.first(where: { $0.id == selectedCandidateID })?.candidateID == candidateID,
                  candidateDetailRequestID == requestID else { return }
            selectedCandidateDetail = candidate
        } catch {
            // The bounded public result remains usable if the optional full
            // audit row is unavailable.  Do not turn the entire report into an
            // error state for a lazy detail failure.
        }
    }

    func loadCandidateFolds(runID: String, candidateID: String) async {
        let requestID = UUID()
        candidateFoldRequestID = requestID
        isLoadingCandidateFolds = true
        candidateFoldError = nil
        defer {
            if candidateFoldRequestID == requestID { isLoadingCandidateFolds = false }
        }
        do {
            let folds = try await engine.candidateFolds(jobID: runID, candidateID: candidateID)
            guard selectedRunID == runID,
                  selectedScienceResult?.auditCandidates.first(where: { $0.id == selectedCandidateID })?.candidateID == candidateID,
                  candidateFoldRequestID == requestID else { return }
            selectedCandidateFolds = folds
        } catch {
            guard selectedRunID == runID, candidateFoldRequestID == requestID else { return }
            candidateFoldError = error.localizedDescription
        }
    }

    func resultDownloadURL(for runID: String) -> URL? { engine.resultDownloadURL(jobID: runID) }
    func auditDownloadURL(for runID: String) -> URL? { engine.auditDownloadURL(jobID: runID) }

    enum RunArtifactKind { case candidatePDF, resultJSON, fullAudit, auditCSV }

    func exportRunArtifact(runID: String, kind: RunArtifactKind) {
        guard !isExportingRunArtifact else { return }
        isExportingRunArtifact = true
        Task {
            defer { isExportingRunArtifact = false }
            do {
                let data: Data
                let suffix: String
                switch kind {
                case .candidatePDF:
                    guard let record = runRecords.first(where: { $0.id == runID }),
                          let result = selectedScienceResult,
                          let summaryCandidate = result.auditCandidates.first(where: { $0.id == selectedCandidateID }) else {
                        throw EngineError.invalidResponse
                    }
                    let candidate: ResultCandidate
                    if let detail = selectedCandidateDetail,
                       detail.candidateID == summaryCandidate.candidateID {
                        candidate = detail
                    } else {
                        candidate = summaryCandidate
                    }
                    data = try ScienceReportPDFExporter.render(
                        run: record,
                        result: result,
                        candidate: candidate,
                        folds: selectedCandidateFolds?.candidateID == candidate.candidateID ? selectedCandidateFolds : nil
                    )
                    suffix = "\(candidate.candidateID)-science-report.pdf"
                case .resultJSON:
                    data = try await engine.resultArtifactData(jobID: runID)
                    suffix = "complete-result.json"
                case .fullAudit:
                    data = try await engine.auditArtifactData(jobID: runID)
                    suffix = "full-audit.json.gz"
                case .auditCSV:
                    guard let selectedScienceResult else { throw EngineError.invalidResponse }
                    data = auditCSV(selectedScienceResult.auditCandidates)
                    suffix = "audit-ledger.csv"
                }

                let panel = NSSavePanel()
                panel.canCreateDirectories = true
                panel.isExtensionHidden = false
                panel.nameFieldStringValue = "Avril-\(runID)-\(suffix)"
                guard panel.runModal() == .OK, let url = panel.url else { return }
                try data.write(to: url, options: .atomic)
            } catch {
                errorBanner = error.localizedDescription
            }
        }
    }

    private func auditCSV(_ candidates: [ResultCandidate]) -> Data {
        let header = ["rank", "candidate_id", "period_days", "epoch_btjd", "duration_hours", "testable_events", "supported_events", "red_noise_mes_like", "event_coherence_snr", "depth_ppm", "gap_status", "physical_status", "pixel_status", "science_priority"]
        var rows: [String] = []
        rows.reserveCapacity(candidates.count)
        for candidate in candidates { rows.append(auditCSVRow(candidate)) }
        let headerLine = header.map(csvCell).joined(separator: ",")
        let body = rows.joined(separator: "\n")
        let document = "\u{FEFF}\(headerLine)\n\(body)\n"
        return Data(document.utf8)
    }

    private func auditCSVRow(_ candidate: ResultCandidate) -> String {
        let gapValue = candidate.gapDiagnostics?["gap_dominated_weak_remainder"]?.boolValue
        let physicalValue = candidate.raw.value(at: "physical_vetting", "passed")?.boolValue
        var values: [String] = []
        values.append(csvInteger(candidate.rank))
        values.append(candidate.candidateID)
        values.append(csvNumber(candidate.periodDays))
        values.append(csvNumber(candidate.epochBTJD))
        values.append(csvNumber(candidate.durationHours))
        values.append(csvInteger(candidate.testableEventCount))
        values.append(csvInteger(candidate.supportedEventCount))
        values.append(csvNumber(candidate.calibratedMES))
        values.append(csvNumber(candidate.eventCoherenceSNR))
        values.append(csvNumber(candidate.depthPPM))
        values.append(gapStatusCSV(gapValue))
        values.append(physicalStatusCSV(physicalValue))
        values.append(candidate.raw.value(at: "pixel_vetting", "aggregate_status")?.stringValue ?? "not_computed")
        values.append(candidate.priority)
        return values.map(csvCell).joined(separator: ",")
    }

    private func csvCell(_ value: String) -> String {
        "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
    }

    private func csvNumber(_ value: Double?) -> String {
        guard let value else { return "" }
        return String(value)
    }

    private func csvInteger(_ value: Int?) -> String {
        guard let value else { return "" }
        return String(value)
    }

    private func gapStatusCSV(_ value: Bool?) -> String {
        if value == true { return "gap_dominated" }
        if value == false { return "checked_clear" }
        return "not_computed"
    }

    private func physicalStatusCSV(_ value: Bool?) -> String {
        if value == true { return "pass" }
        if value == false { return "risk" }
        return "not_computed"
    }

    func addSelectedToQueue(runImmediately: Bool = false) {
        guard let target = selectedTarget else { return }
        if tasks.contains(where: { $0.target.id == target.id && [.queued, .running].contains($0.state) }) {
            errorBanner = "TIC \(target.ticID) is already queued."
            return
        }
        let task = ScienceTask(
            id: UUID(),
            target: target,
            state: .queued,
            progress: 0,
            stage: "queued",
            message: "Waiting for the local science engine",
            events: [],
            createdAt: Date()
        )
        tasks.append(task)
        if runImmediately || !tasks.contains(where: { $0.state == .running }) {
            startQueueIfNeeded()
        }
    }

    func startQueueIfNeeded() {
        guard runner == nil else { return }
        runner = Task { [weak self] in
            guard let self else { return }
            while let next = self.tasks.first(where: { $0.state == .queued }) {
                if Task.isCancelled { break }
                await self.execute(taskID: next.id)
            }
            self.runner = nil
        }
    }

    func stop(task: ScienceTask) {
        guard let jobID = task.jobID, let token = task.controlToken else {
            update(task.id) { $0.state = .cancelled; $0.message = "Removed from queue" }
            return
        }
        Task {
            do {
                _ = try await engine.cancel(jobID: jobID, token: token)
                update(task.id) { $0.state = .cancelled; $0.message = "Cancelled" }
            } catch {
                errorBanner = error.localizedDescription
            }
        }
    }

    func shutdown() {
        runner?.cancel()
        healthMonitor?.cancel()
        engine.stop()
    }

    private func startHealthMonitor() {
        guard healthMonitor == nil else { return }
        healthMonitor = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard let self, !Task.isCancelled else { return }
                await self.recoverEngine(force: false)
            }
        }
    }

    private func recoverEngine(force: Bool) async {
        if !force, let version = await engine.currentVersionIfHealthy() {
            consecutiveHealthFailures = 0
            if case .ready = engineState {} else { engineState = .ready(version: version) }
            return
        }
        if !force {
            consecutiveHealthFailures += 1
            guard consecutiveHealthFailures >= 2 else { return }
        }
        guard !isRecoveringEngine else { return }
        isRecoveringEngine = true
        engineState = .starting
        engine.stop()
        do {
            let version = try await engine.start()
            consecutiveHealthFailures = 0
            engineState = .ready(version: version)
            errorBanner = nil
            if targets.isEmpty { await loadTargets() }
            await loadRunRecords(loadSelection: section == .review && selectedScienceResult == nil)
        } catch {
            engineState = .failed(error.localizedDescription)
            errorBanner = t("Local science engine: \(error.localizedDescription)", "本地科学引擎：\(error.localizedDescription)")
        }
        isRecoveringEngine = false
    }

    private func execute(taskID: UUID) async {
        guard let target = tasks.first(where: { $0.id == taskID })?.target else { return }
        // Never label one target's cached samples with a newly selected row.
        // The light-curve viewport follows the executing task, while catalog
        // selection remains free to configure a later queue entry.
        lightCurve = nil
        lightCurveTaskID = taskID
        update(taskID) {
            $0.state = .running
            $0.stage = "starting"
            $0.message = "Preparing TIC \(target.ticID)"
            $0.progress = 0.01
        }
        do {
            let created = try await engine.createJob(
                ticID: target.ticID,
                source: source,
                pixels: includePixelFollowup
            )
            update(taskID) {
                $0.jobID = created.id
                $0.controlToken = created.controlToken
            }

            var lastLightCurveCheck = Date.distantPast
            while true {
                try Task.checkCancellation()
                let job = try await engine.job(id: created.id)
                update(taskID) {
                    $0.progress = min(max((job.percent ?? 0) / 100, 0.02), 1)
                    $0.stage = job.stage ?? job.status
                    $0.message = job.message ?? "Avril V3 is processing this target"
                    $0.candidateCount = job.result?.candidateCount
                    $0.runtimeSeconds = job.result?.runtimeSeconds
                    if let events = job.events { $0.events = events }
                }

                if Date().timeIntervalSince(lastLightCurveCheck) > 4 {
                    lastLightCurveCheck = Date()
                    if let payload = try? await engine.lightCurve(jobID: created.id), payload.status == "ready" {
                        lightCurve = payload
                        lightCurveTaskID = taskID
                    }
                }

                switch job.status {
                case "complete":
                    update(taskID) {
                        $0.state = .complete
                        $0.progress = 1
                        $0.stage = "complete"
                        $0.message = job.message ?? "Evidence report ready"
                    }
                    if let payload = try? await engine.lightCurve(jobID: created.id), payload.status == "ready" {
                        lightCurve = payload
                        lightCurveTaskID = taskID
                    }
                    await loadRunRecords()
                    return
                case "cancelled":
                    update(taskID) { $0.state = .cancelled }
                    await loadRunRecords()
                    return
                case "failed", "error":
                    update(taskID) { $0.state = .failed; $0.message = job.message ?? "Science run failed" }
                    await loadRunRecords()
                    return
                default:
                    try await Task.sleep(for: .seconds(2))
                }
            }
        } catch is CancellationError {
            update(taskID) { $0.state = .cancelled; $0.message = "Cancelled" }
        } catch {
            update(taskID) { $0.state = .failed; $0.message = error.localizedDescription }
            errorBanner = "TIC \(target.ticID): \(error.localizedDescription)"
        }
    }

    private func update(_ id: UUID, mutation: (inout ScienceTask) -> Void) {
        guard let index = tasks.firstIndex(where: { $0.id == id }) else { return }
        mutation(&tasks[index])
    }
}
