import AppKit
import CoreGraphics
import Foundation

/// Native, deterministic scientific-report renderer for Avril candidate evidence.
///
/// The exporter intentionally uses AppKit/Core Graphics rather than HTML or a
/// web view.  It only renders values present in the sealed run payload.  A
/// missing measurement is written as `NOT COMPUTED`; it is never inferred from
/// another statistic and is never rendered as a pass.
enum ScienceReportPDFExporter {
    enum ExportError: LocalizedError {
        case cannotCreateConsumer
        case cannotCreateContext

        var errorDescription: String? {
            switch self {
            case .cannotCreateConsumer: "Could not create the PDF data consumer."
            case .cannotCreateContext: "Could not create the PDF graphics context."
            }
        }
    }

    /// Produce a complete A4-landscape scientific candidate report.
    ///
    /// - Parameters:
    ///   - run: Persistent execution metadata from the desktop bridge.
    ///   - result: Sealed scientific result document for the run.
    ///   - candidate: Exact audit-ledger candidate to report.
    ///   - folds: Optional full-cadence, locked-ephemeris fold document.  When
    ///     absent, the fold section explicitly reports `NOT COMPUTED`.
    ///   - generatedAt: Timestamp used in the report footer; injectable for
    ///     reproducible tests.
    static func render(
        run: RunRecordSummary,
        result: ScienceResultDocument,
        candidate: ResultCandidate,
        folds: CandidateFoldDocument? = nil,
        generatedAt: Date = Date()
    ) throws -> Data {
        let storage = NSMutableData()
        guard let consumer = CGDataConsumer(data: storage as CFMutableData) else {
            throw ExportError.cannotCreateConsumer
        }
        var mediaBox = PDFCanvas.pageRect
        guard let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            throw ExportError.cannotCreateContext
        }

        let canvas = PDFCanvas(
            context: context,
            runID: run.id,
            candidateID: candidate.candidateID,
            generatedAt: generatedAt
        )
        ReportRenderer(
            canvas: canvas,
            run: run,
            result: result,
            candidate: candidate,
            folds: folds
        ).draw()
        context.closePDF()
        return storage as Data
    }

    /// Alias suited to call sites that name data-producing functions explicitly.
    static func makeData(
        run: RunRecordSummary,
        result: ScienceResultDocument,
        candidate: ResultCandidate,
        folds: CandidateFoldDocument? = nil,
        generatedAt: Date = Date()
    ) throws -> Data {
        try render(run: run, result: result, candidate: candidate, folds: folds, generatedAt: generatedAt)
    }
}

// MARK: - Report composition

private struct ReportRenderer {
    let canvas: PDFCanvas
    let run: RunRecordSummary
    let result: ScienceResultDocument
    let candidate: ResultCandidate
    let folds: CandidateFoldDocument?

    func draw() {
        drawExecutiveSummary()
        drawScientificScope()
        drawFoldEvidence()
        drawGapAndEvents()
        drawVettingMatrix()
        drawReproducibility()
    }

    // MARK: Page 1

    private func drawExecutiveSummary() {
        canvas.beginPage(section: "CANDIDATE REPORT", title: "Scientific evidence summary")

        canvas.text(
            "AVRIL V3 / EXOPLANET TRANSIT SEARCH",
            in: CGRect(x: 34, y: 76, width: 480, height: 18),
            style: .eyebrow,
            color: Palette.cyan
        )
        canvas.text(
            candidate.candidateID,
            in: CGRect(x: 34, y: 94, width: 500, height: 34),
            style: .display,
            color: Palette.navy
        )
        canvas.text(
            "Target \(targetLabel)  /  audit rank \(value(candidate.rank))  /  job \(run.id)",
            in: CGRect(x: 34, y: 130, width: 570, height: 18),
            style: .body,
            color: Palette.slate
        )

        let conclusion = conclusionEvidence
        canvas.statusBanner(
            frame: CGRect(x: 600, y: 78, width: 208, height: 70),
            title: conclusion.title,
            detail: conclusion.detail,
            tone: conclusion.tone
        )

        let metrics: [(String, String, String)] = [
            ("PERIOD", decimal(candidate.periodDays, digits: 8, suffix: " d"), "locked ephemeris"),
            ("EPOCH", decimal(candidate.epochBTJD, digits: 6, suffix: " BTJD"), "transit center"),
            ("DURATION", decimal(candidate.durationHours, digits: 3, suffix: " h"), "fitted duration"),
            ("DEPTH", decimal(candidate.depthPPM, digits: 1, suffix: " ppm"), "weighted depth"),
            ("RED-NOISE MES", decimal(candidate.calibratedMES, digits: 3), "null calibrated"),
            ("PHASE-NULL FAP", scientific(candidate.phaseNullFAP), "local empirical"),
            ("LOOK-ELSEWHERE FAP", scientific(candidate.lookElsewhereFAP), "global estimate"),
            ("EVENT SUPPORT", supportLabel, "supported / testable"),
        ]
        for (index, metric) in metrics.enumerated() {
            let column = index % 4
            let row = index / 4
            canvas.metricCard(
                frame: CGRect(x: 34 + CGFloat(column) * 194, y: 166 + CGFloat(row) * 83, width: 184, height: 72),
                label: metric.0,
                value: metric.1,
                note: metric.2
            )
        }

        canvas.sectionTitle("Automated disposition", y: 345)
        let priority = normalized(candidate.priority)
        let disposition = normalized(candidate.disposition)
        canvas.keyValueGrid(
            rows: [
                ("Science priority", priority),
                ("Meaningful-candidate gate", disposition),
                ("Significance status", normalized(candidate.significanceStatus)),
                ("Passes 7.1 reference", boolean(candidate.passesReference)),
                ("Holdout reproduction", boolean(candidate.raw.value(at: "holdout_verification", "passed")?.boolValue)),
                ("Confirmation required", boolean(candidate.confirmationRequired)),
            ],
            frame: CGRect(x: 34, y: 370, width: 500, height: 112),
            columns: 2
        )

        canvas.sectionTitle("Recorded risks and interpretation", y: 345, x: 558, width: 250)
        let risks = candidate.riskFlags
        canvas.text(
            risks.isEmpty ? "No automated risk flag was recorded. This does not confirm a planet." : risks.map { "• \(normalized($0))" }.joined(separator: "\n"),
            in: CGRect(x: 558, y: 370, width: 250, height: 50),
            style: .small,
            color: risks.isEmpty ? Palette.slate : Palette.red
        )
        canvas.text(
            candidate.explanation,
            in: CGRect(x: 558, y: 425, width: 250, height: 57),
            style: .caption,
            color: Palette.slate
        )

        canvas.notice(
            "SCIENTIFIC SCOPE  This document reports automated transit-search evidence. It is not a planet validation, confirmation, catalog-novelty determination, or substitute for pixel-level and astrophysical follow-up.",
            frame: CGRect(x: 34, y: 500, width: 774, height: 40)
        )
        canvas.endPage()
    }

    // MARK: Page 2

    private func drawScientificScope() {
        canvas.beginPage(section: "METHOD / SCOPE", title: "Search provenance and evidence funnel")

        canvas.sectionTitle("Run and target scope", y: 78)
        canvas.keyValueGrid(
            rows: [
                ("Target", targetLabel),
                ("TIC identifier", result.ticID.map(String.init) ?? "NOT COMPUTED"),
                ("Observed sectors", list(result.segments)),
                ("Sector count", result.segments.isEmpty ? "NOT COMPUTED" : String(result.segments.count)),
                ("Run status", normalized(run.status)),
                ("Scientific status", optional(result.scientificStatus)),
                ("Source mode", optional(run.source)),
                ("Pixel follow-up", optional(run.pixels)),
            ],
            frame: CGRect(x: 34, y: 103, width: 482, height: 143),
            columns: 2
        )

        canvas.sectionTitle("Execution timing", y: 78, x: 540, width: 268)
        canvas.keyValueGrid(
            rows: [
                ("Queued", seconds(run.queueSeconds)),
                ("Compute", seconds(run.computeSeconds)),
                ("End to end", seconds(run.totalSeconds)),
                ("Started", optional(run.startedAt)),
                ("Completed", optional(run.completedAt)),
            ],
            frame: CGRect(x: 540, y: 103, width: 268, height: 143),
            columns: 1
        )

        canvas.sectionTitle("Evidence funnel", y: 270)
        let funnel: [(String, Int?)] = [
            ("RAW SEARCH HYPOTHESES", result.rawHypothesisCount),
            ("GLOBALLY STABLE FAMILIES", result.globallyStableFamilyCount),
            ("DEDUPLICATED FAMILIES", result.deduplicatedFamilyCount),
            ("CANONICAL CANDIDATES", result.canonicalCandidateCount),
            ("AUDIT LEDGER ROWS", result.auditCandidates.isEmpty ? nil : result.auditCandidates.count),
        ]
        canvas.funnel(funnel, frame: CGRect(x: 34, y: 297, width: 482, height: 196))

        canvas.sectionTitle("Search provenance", y: 270, x: 540, width: 268)
        canvas.keyValueGrid(
            rows: [
                ("Source methods", candidate.sourceMethods.isEmpty ? "NOT COMPUTED" : candidate.sourceMethods.joined(separator: " · ")),
                ("Discovery blocks", candidate.discoveredBlocks.isEmpty ? "NOT COMPUTED" : candidate.discoveredBlocks.map(String.init).joined(separator: ", ")),
                ("Discovery sectors", setList(candidate.discoverySectors)),
                ("Holdout sectors", setList(candidate.holdoutSectors)),
                ("Source strength", decimal(candidate.sourceStrength, digits: 5)),
                ("Ensemble score", decimal(candidate.ensembleScore, digits: 5)),
                ("Direct harmonic alias", boolean(candidate.directHarmonicAlias)),
            ],
            // Seven provenance rows need a true 28-point cell height at the
            // report font size.  The previous 172-point frame compressed the
            // label/value baselines until adjacent rows visibly overlapped.
            frame: CGRect(x: 540, y: 297, width: 268, height: 196),
            columns: 1
        )

        canvas.notice(
            "Interpretation rule: ranking scores prioritize review; they are not calibrated planet probabilities. Discovery and holdout partitions are shown separately wherever the sealed payload provides them.",
            frame: CGRect(x: 34, y: 508, width: 774, height: 36)
        )
        canvas.endPage()
    }

    // MARK: Page 3

    private func drawFoldEvidence() {
        canvas.beginPage(section: "FOLD / BLOCKS", title: "Locked-ephemeris phase evidence")

        guard let folds else {
            canvas.unavailable(
                title: "FULL-CADENCE PHASE FOLD — NOT COMPUTED",
                detail: "No full-cadence fold document was supplied to this export. No preview-decimated light curve has been substituted.",
                frame: CGRect(x: 34, y: 90, width: 774, height: 170)
            )
            canvas.sectionTitle("Partition recorded in candidate audit", y: 294)
            canvas.keyValueGrid(
                rows: [
                    ("Discovery sectors", setList(candidate.discoverySectors)),
                    ("Holdout sectors", setList(candidate.holdoutSectors)),
                    ("Period", decimal(candidate.periodDays, digits: 10, suffix: " d")),
                    ("Epoch", decimal(candidate.epochBTJD, digits: 8, suffix: " BTJD")),
                    ("Duration", decimal(candidate.durationHours, digits: 5, suffix: " h")),
                ],
                frame: CGRect(x: 34, y: 320, width: 500, height: 132),
                columns: 2
            )
            canvas.notice(
                "NOT COMPUTED means the measurement was unavailable to this report; it is not a negative result and is not a pass.",
                frame: CGRect(x: 34, y: 495, width: 774, height: 45)
            )
            canvas.endPage()
            return
        }

        let locked = folds.ephemerisLocked ? "LOCKED — NO REFIT" : "NOT LOCKED"
        canvas.pill(locked, x: 34, y: 78, tone: folds.ephemerisLocked ? .good : .warning)
        canvas.text(
            "Every plotted bin is aggregated from quality-clean FITS samples at the sealed period and epoch. Discovery and holdout evidence remain separated.",
            in: CGRect(x: 175, y: 79, width: 633, height: 28),
            style: .caption,
            color: Palette.slate
        )

        canvas.foldPlot(
            title: "DISCOVERY PARTITION",
            series: folds.folds.discovery,
            frame: CGRect(x: 34, y: 120, width: 374, height: 236),
            color: Palette.cyan,
            durationHours: folds.durationHours,
            periodDays: folds.periodDays
        )
        canvas.foldPlot(
            title: "HOLDOUT PARTITION",
            series: folds.folds.holdout,
            frame: CGRect(x: 434, y: 120, width: 374, height: 236),
            color: Palette.green,
            durationHours: folds.durationHours,
            periodDays: folds.periodDays
        )

        canvas.sectionTitle("Full-cadence sample accounting", y: 378)
        canvas.keyValueGrid(
            rows: [
                ("Files resolved", String(folds.samples.sourceFilesResolved)),
                ("Files used", String(folds.samples.sourceFilesUsed)),
                ("Files skipped", String(folds.samples.sourceFilesSkipped)),
                ("Quality-clean samples", String(folds.samples.qualityCleanTotal)),
                ("In-window samples", String(folds.samples.windowTotal)),
                ("Phase window", String(format: "%.5f … %.5f cycles", folds.phaseWindow.minimumCycles, folds.phaseWindow.maximumCycles)),
                ("Discovery sectors", list(folds.folds.discovery.sectors)),
                ("Holdout sectors", list(folds.folds.holdout.sectors)),
            ],
            frame: CGRect(x: 34, y: 404, width: 774, height: 116),
            columns: 4
        )
        canvas.endPage()
    }

    // MARK: Pages 4…n

    private func drawGapAndEvents() {
        let rows = candidate.eventRows
        // Ten rows also fit on the first page below the full gap summary; use
        // the same bound on continuation pages so pagination is deterministic.
        let pageSize = 10
        let pages = max(1, Int(ceil(Double(rows.count) / Double(pageSize))))

        for pageIndex in 0..<pages {
            canvas.beginPage(
                section: "EVENT AUDIT",
                title: pages == 1 ? "Gap-boundary diagnostics and event evidence" : "Gap-boundary diagnostics and event evidence — \(pageIndex + 1)/\(pages)"
            )
            if pageIndex == 0 { drawGapSummary() }
            let tableY: CGFloat = pageIndex == 0 ? 252 : 88
            let lower = pageIndex * pageSize
            let upper = min(rows.count, lower + pageSize)
            let subset = lower < upper ? Array(rows[lower..<upper]) : []
            drawEventTable(subset, y: tableY, startingIndex: lower)

            if rows.isEmpty {
                canvas.unavailable(
                    title: "PER-EVENT GAP AUDIT — NOT COMPUTED",
                    detail: "The sealed candidate payload contains no per-event diagnostic rows.",
                    frame: CGRect(x: 34, y: tableY + 42, width: 774, height: 105)
                )
            }
            canvas.endPage()
        }
    }

    private func drawGapSummary() {
        guard let gap = candidate.gapDiagnostics else {
            canvas.unavailable(
                title: "GAP DIAGNOSTICS — NOT COMPUTED",
                detail: "No event-gap diagnostic object is present. Absence is not interpreted as clean data.",
                frame: CGRect(x: 34, y: 82, width: 774, height: 138)
            )
            return
        }

        let critical = gap["critical_event_count"]?.intValue
        let clean = gap["uncompromised_event_count"]?.intValue
        let supportedClean = gap["supported_uncompromised_events"]?.intValue
        let dominantSector = gap["dominant_event_segment"]?.intValue
        let dominantSES = gap["dominant_event_ses_like"]?.doubleValue
        let remainderMES = gap.value(at: "combined_without_gap_compromised_events", "mes_like")?.doubleValue
        let dominated = gap["gap_dominated_weak_remainder"]?.boolValue

        canvas.statusBanner(
            frame: CGRect(x: 34, y: 82, width: 220, height: 112),
            title: dominated == true ? "GAP-DOMINATED" : "GAP AUDIT AVAILABLE",
            detail: dominated == true
                ? "A compromised dominant event lifts an otherwise weak event train."
                : "Per-event coverage and long-gap proximity were measured.",
            tone: dominated == true ? .bad : .neutral
        )
        canvas.keyValueGrid(
            rows: [
                ("Critical events", value(critical)),
                ("Uncompromised events", value(clean)),
                ("Supported clean events", value(supportedClean)),
                ("Dominant sector", dominantSector.map { "Sector \($0)" } ?? "NOT COMPUTED"),
                ("Dominant SES-like", decimal(dominantSES, digits: 3)),
                ("MES without compromised events", decimal(remainderMES, digits: 3)),
            ],
            frame: CGRect(x: 276, y: 82, width: 532, height: 112),
            columns: 3
        )
        canvas.text(
            gap["policy"]?.stringValue ?? "No gap-diagnostic policy was recorded.",
            in: CGRect(x: 34, y: 208, width: 774, height: 26),
            style: .caption,
            color: Palette.slate
        )
    }

    private func drawEventTable(_ rows: [JSONValue], y: CGFloat, startingIndex: Int) {
        let columns: [PDFCanvas.TableColumn] = [
            .init("#", 28, .center),
            .init("SECTOR", 54, .center),
            .init("CYCLE", 44, .center),
            .init("CENTER BTJD", 83, .right),
            .init("SES", 52, .right),
            .init("COVERAGE", 63, .right),
            .init("LEFT / RIGHT", 78, .center),
            .init("SYMMETRY", 64, .right),
            .init("GAP EDGE", 68, .right),
            .init("CONTAINING", 67, .right),
            .init("STATUS", 88, .center),
        ]
        canvas.tableHeader(columns: columns, x: 34, y: y)
        for (offset, row) in rows.enumerated() {
            let compromised = row["critical_gap_compromise"]?.boolValue
            let values = [
                String(startingIndex + offset + 1),
                value(row["segment"]?.intValue),
                value(row["cycle"]?.intValue),
                decimal(row["center_btjd"]?.doubleValue, digits: 5),
                decimal(row["ses_like"]?.doubleValue, digits: 2),
                percent(row["transit_coverage_fraction"]?.doubleValue),
                "\(value(row["left_baseline_points"]?.intValue)) / \(value(row["right_baseline_points"]?.intValue))",
                percent(row["baseline_symmetry_fraction"]?.doubleValue),
                decimal(row["nearest_long_gap_edge_hours"]?.doubleValue, digits: 2, suffix: " h"),
                decimal(row["containing_long_gap_hours"]?.doubleValue, digits: 2, suffix: " h"),
                compromised == true ? "COMPROMISED" : compromised == false ? "CLEAR" : "NOT COMPUTED",
            ]
            canvas.tableRow(
                values: values,
                columns: columns,
                x: 34,
                y: y + 26 + CGFloat(offset) * 25,
                tone: compromised == true ? .bad : compromised == false ? .good : .neutral,
                alternate: offset.isMultiple(of: 2)
            )
        }
    }

    // MARK: Vetting

    private func drawVettingMatrix() {
        canvas.beginPage(section: "VETTING", title: "False-positive and evidence-state matrix")
        canvas.text(
            "A missing check is explicitly NOT COMPUTED. It is never treated as a clear result.",
            in: CGRect(x: 34, y: 78, width: 774, height: 20),
            style: .body,
            color: Palette.slate
        )

        let checks = vettingChecks
        for (index, check) in checks.enumerated() {
            let column = index % 2
            let row = index / 2
            canvas.vettingCard(
                frame: CGRect(x: 34 + CGFloat(column) * 397, y: 110 + CGFloat(row) * 58, width: 377, height: 50),
                title: check.title,
                status: check.status,
                detail: check.detail,
                tone: check.tone
            )
        }
        canvas.endPage()
    }

    private var vettingChecks: [VettingCheck] {
        let physical = candidate.raw.value(at: "physical_vetting")
        let reference = candidate.raw.value(at: "reference_vetting")
        let pixelStatus = candidate.raw.value(at: "pixel_vetting", "aggregate_status")?.stringValue
        let significanceAvailable = !(candidate.significanceStatus.hasPrefix("skipped") || ["not_computed", "insufficient_statistics"].contains(candidate.significanceStatus))

        var checks: [VettingCheck] = [
            .init(
                title: "Red noise / phase null",
                status: significanceAvailable ? (candidate.passesReference == true ? "PASS" : "FAIL") : "NOT COMPUTED",
                detail: significanceAvailable
                    ? "\(normalized(candidate.significanceStatus)); MES \(decimal(candidate.calibratedMES, digits: 3)); local FAP \(scientific(candidate.phaseNullFAP))"
                    : "No decisive calibrated significance result is present.",
                tone: significanceAvailable ? (candidate.passesReference == true ? .good : .bad) : .neutral
            ),
            boolCheck(
                title: "Fixed-ephemeris reproduction",
                value: candidate.raw.value(at: "holdout_verification", "passed")?.boolValue,
                detail: optional(candidate.raw.value(at: "holdout_verification", "outcome")?.stringValue)
            ),
            gapCheck,
            physicalSummary(physical),
            .init(
                title: "Pixel localization",
                status: pixelStatus.map(normalized) ?? "NOT COMPUTED",
                detail: candidate.raw.value(at: "pixel_vetting", "sectors_available")?.intValue.map { "\($0) sector products available" } ?? "No pixel-localization measurement is present.",
                tone: ["on_target", "consistent_on_target"].contains(pixelStatus ?? "") ? .good : (pixelStatus?.contains("off_target") == true ? .bad : .neutral)
            ),
            referenceSummary(reference),
        ]
        checks.append(physicalCheck(physical, key: "odd_even", title: "Odd / even depth") { value in
            "difference \(decimal(value["difference_sigma"]?.doubleValue, digits: 2, suffix: " σ")); odd \(decimal(value["odd_depth_ppm"]?.doubleValue, digits: 1, suffix: " ppm")); even \(decimal(value["even_depth_ppm"]?.doubleValue, digits: 1, suffix: " ppm"))"
        })
        checks.append(physicalCheck(physical, key: "secondary_eclipse", title: "Secondary eclipse") { value in
            "S/N \(decimal(value["snr"]?.doubleValue, digits: 2)); depth ratio \(decimal(value["depth_ratio"]?.doubleValue, digits: 3))"
        })
        checks.append(physicalCheck(physical, key: "v_shape", title: "Transit shape") { value in
            "ingress fraction \(decimal(value["ingress_fraction"]?.doubleValue, digits: 3)); Δχ² vs U \(decimal(value["delta_chi2_vs_u"]?.doubleValue, digits: 2))"
        })
        checks.append(physicalCheck(physical, key: "centroid_motion", title: "Centroid motion") { value in
            "shift \(decimal(value["shift_pixels"]?.doubleValue, digits: 6, suffix: " px")); significance \(decimal(value["significance"]?.doubleValue, digits: 2, suffix: " σ"))"
        })
        checks.append(physicalCheck(physical, key: "neighbor_contamination", title: "Neighbor contamination") { value in
            "CROWDSAP \(decimal(value["crowdsap"]?.doubleValue, digits: 4)); required eclipse \(percent(value["required_neighbor_eclipse_fraction"]?.doubleValue)); mimic \(boolean(value["neighbor_can_mimic"]?.boolValue))"
        })
        checks.append(referenceModelShift(reference))
        checks.append(referenceCoverage(reference))
        checks.append(referenceVariability(reference))
        return checks
    }

    // MARK: Reproducibility

    private func drawReproducibility() {
        canvas.beginPage(section: "AUDIT / REPRODUCIBILITY", title: "Exact inputs, methods and caveats")

        canvas.sectionTitle("Immutable identifiers", y: 78)
        canvas.keyValueGrid(
            rows: [
                ("Job ID", run.id),
                ("Candidate ID", candidate.candidateID),
                ("Target", targetLabel),
                ("Release ID", optional(result.releaseID)),
                ("Engine version", optional(result.engineVersion ?? run.engine)),
                ("Production backend", optional(result.productionSignificanceBackend ?? run.backend)),
                ("Created", optional(run.createdAt)),
                ("Updated", optional(run.updatedAt)),
            ],
            frame: CGRect(x: 34, y: 103, width: 774, height: 117),
            columns: 4
        )

        canvas.sectionTitle("Candidate ephemeris and audit fields", y: 242)
        canvas.keyValueGrid(
            rows: [
                ("Period days", decimal(candidate.periodDays, digits: 10)),
                ("Epoch BTJD", decimal(candidate.epochBTJD, digits: 8)),
                ("Duration hours", decimal(candidate.durationHours, digits: 5)),
                ("Weighted depth ppm", decimal(candidate.depthPPM, digits: 4)),
                ("Null-calibrated MES", decimal(candidate.calibratedMES, digits: 5)),
                ("Raw combined MES", decimal(candidate.rawCombinedMES, digits: 5)),
                ("Event coherence S/N", decimal(candidate.eventCoherenceSNR, digits: 5)),
                ("Phase-null local FAP", scientific(candidate.phaseNullFAP)),
                ("Look-elsewhere FAP", scientific(candidate.lookElsewhereFAP)),
                ("Significance events", value(candidate.significanceEventCount)),
                ("Testable events", value(candidate.testableEventCount)),
                ("Supported events", value(candidate.supportedEventCount)),
            ],
            frame: CGRect(x: 34, y: 268, width: 774, height: 116),
            columns: 4
        )

        canvas.sectionTitle("Fold method", y: 405)
        let methodRows: [(String, String)]
        if let method = folds?.method {
            methodRows = [
                ("Time basis", method.timeBasis),
                ("Quality filter", method.qualityFilter),
                ("Flux selection", method.fluxSelection),
                ("Normalization", method.normalization),
                ("Phase definition", method.phaseDefinition),
                ("Aggregation", method.aggregation),
                ("Uncertainty", method.uncertainty),
                ("Decimation", method.decimation),
                ("Ephemeris refit", method.ephemerisRefit),
            ]
        } else {
            methodRows = [("Full-cadence fold method", "NOT COMPUTED")]
        }
        canvas.keyValueGrid(
            rows: methodRows,
            frame: CGRect(x: 34, y: 430, width: 774, height: 84),
            columns: 3
        )

        canvas.notice(
            "REPRODUCIBILITY NOTE  Values are copied from the selected sealed audit row and optional locked-fold product. No missing statistic is reconstructed. Re-run against the same release, source FITS products and recorded ephemeris before external use.",
            frame: CGRect(x: 34, y: 520, width: 774, height: 28)
        )
        canvas.endPage()
    }

    // MARK: Evidence interpretation helpers

    private var targetLabel: String {
        if let tic = result.ticID { return "TIC \(tic)" }
        if let target = result.target, !target.isEmpty { return target }
        return run.target.isEmpty ? "NOT COMPUTED" : run.target
    }

    private var supportLabel: String {
        guard let testable = candidate.testableEventCount ?? candidate.eventCount else { return "NOT COMPUTED" }
        guard let supported = candidate.supportedEventCount else { return "\(testable) testable" }
        if let fraction = candidate.eventSupportFraction {
            return "\(supported) / \(testable) (\(Int((fraction * 100).rounded()))%)"
        }
        return "\(supported) / \(testable)"
    }

    private var conclusionEvidence: (title: String, detail: String, tone: ReportTone) {
        if candidate.gapDiagnostics?["gap_dominated_weak_remainder"]?.boolValue == true {
            return ("DOWNGRADED / GAP RISK", "The strongest event is gap-compromised and the clean remainder is weak.", .bad)
        }
        if candidate.passesReference == true {
            return ("REFERENCE PASSED", "Automated threshold passed; pixel, novelty and astrophysical validation remain required.", .good)
        }
        if candidate.passesReference == false {
            return ("REFERENCE NOT PASSED", "This audit row did not pass the recorded calibrated significance reference.", .warning)
        }
        return ("DECISION NOT COMPUTED", "No decisive calibrated reference result is present in the sealed payload.", .neutral)
    }

    private var gapCheck: VettingCheck {
        guard let gap = candidate.gapDiagnostics else {
            return .init(title: "Gap support", status: "NOT COMPUTED", detail: "No gap diagnostic object is present.", tone: .neutral)
        }
        if gap["gap_dominated_weak_remainder"]?.boolValue == true {
            return .init(
                title: "Gap support",
                status: "GAP-DOMINATED",
                detail: "\(value(gap["critical_event_count"]?.intValue)) compromised; clean MES \(decimal(gap.value(at: "combined_without_gap_compromised_events", "mes_like")?.doubleValue, digits: 3))",
                tone: .bad
            )
        }
        return .init(
            title: "Gap support",
            status: "MEASURED",
            detail: "\(value(gap["critical_event_count"]?.intValue)) compromised; \(value(gap["uncompromised_event_count"]?.intValue)) uncompromised",
            tone: .good
        )
    }

    private func physicalSummary(_ physical: JSONValue?) -> VettingCheck {
        guard let physical else {
            return .init(title: "Physical false-positive checks", status: "NOT COMPUTED", detail: "No physical-vetting object is present.", tone: .neutral)
        }
        let vetoes = physical["vetoes"]?.arrayValue?.compactMap(\.stringValue) ?? []
        if !vetoes.isEmpty {
            return .init(title: "Physical false-positive checks", status: "RISK", detail: vetoes.map(normalized).joined(separator: " · "), tone: .bad)
        }
        if physical["passed"]?.boolValue == true {
            return .init(title: "Physical false-positive checks", status: "PASS", detail: optional(physical["tier"]?.stringValue), tone: .good)
        }
        return .init(title: "Physical false-positive checks", status: "NOT COMPUTED", detail: "No decisive physical-vetting outcome is present.", tone: .neutral)
    }

    private func referenceSummary(_ reference: JSONValue?) -> VettingCheck {
        guard let reference, reference["available"]?.boolValue == true else {
            return .init(title: "Reference vetter", status: "NOT COMPUTED", detail: "Reference diagnostic suite unavailable.", tone: .neutral)
        }
        let failures = reference["failures"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let warnings = reference["warnings"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let advisories = reference["advisories"]?.arrayValue?.compactMap(\.stringValue) ?? []
        let detail = (failures + warnings + advisories).map(normalized).joined(separator: " · ")
        return .init(
            title: "Reference vetter",
            status: normalized(reference["status"]?.stringValue ?? (failures.isEmpty ? "available" : "failed")),
            detail: detail.isEmpty ? "No recorded warning, advisory or failure." : detail,
            tone: !failures.isEmpty ? .bad : (!warnings.isEmpty || !advisories.isEmpty ? .warning : .good)
        )
    }

    private func physicalCheck(
        _ physical: JSONValue?,
        key: String,
        title: String,
        detail: (JSONValue) -> String
    ) -> VettingCheck {
        guard let item = physical?[key], item["available"]?.boolValue == true else {
            return .init(title: title, status: "NOT COMPUTED", detail: "Measurement unavailable; not a clear result.", tone: .neutral)
        }
        let veto = item["veto"]?.boolValue
        return .init(
            title: title,
            status: veto == true ? "RISK" : veto == false ? "CLEAR" : "MEASURED",
            detail: detail(item),
            tone: veto == true ? .bad : veto == false ? .good : .neutral
        )
    }

    private func referenceModelShift(_ reference: JSONValue?) -> VettingCheck {
        guard let item = reference?["model_shift"], item["available"]?.boolValue == true else {
            return .init(title: "Model-shift diagnostic", status: "NOT COMPUTED", detail: "No model-shift measurement is present.", tone: .neutral)
        }
        return .init(
            title: "Model-shift diagnostic",
            status: "MEASURED",
            detail: "primary \(decimal(item["primary_dip_sigma"]?.doubleValue, digits: 2, suffix: " σ")); secondary \(decimal(item["secondary_dip_sigma"]?.doubleValue, digits: 2, suffix: " σ")); red β \(decimal(item["red_noise_factor"]?.doubleValue, digits: 2))",
            tone: .neutral
        )
    }

    private func referenceCoverage(_ reference: JSONValue?) -> VettingCheck {
        guard let item = reference?["transit_phase_coverage"], item["available"]?.boolValue == true else {
            return .init(title: "Transit phase coverage", status: "NOT COMPUTED", detail: "No phase-coverage measurement is present.", tone: .neutral)
        }
        return .init(
            title: "Transit phase coverage",
            status: percent(item["fraction"]?.doubleValue),
            detail: "\(value(item["occupied_bins"]?.intValue)) / \(value(item["total_bins"]?.intValue)) phase bins occupied",
            tone: .good
        )
    }

    private func referenceVariability(_ reference: JSONValue?) -> VettingCheck {
        guard let item = reference?["sinusoidal_variability"], item["available"]?.boolValue == true else {
            return .init(title: "Sinusoidal variability", status: "NOT COMPUTED", detail: "No variability diagnostic is present.", tone: .neutral)
        }
        let fits = item["fits"]?.arrayValue ?? []
        let strongest = fits.compactMap { $0["sigma"]?.doubleValue }.max()
        let warning = item["warning_sigma"]?.doubleValue
        let isWarning = strongest != nil && warning != nil && strongest! >= warning!
        return .init(
            title: "Sinusoidal variability",
            status: strongest.map { "MAX \(String(format: "%.2f", $0)) σ" } ?? "MEASURED",
            detail: "\(fits.count) timescales; \(value(item["cadences_used"]?.intValue)) cadences",
            tone: isWarning ? .warning : .good
        )
    }

    private func boolCheck(title: String, value: Bool?, detail: String) -> VettingCheck {
        .init(
            title: title,
            status: value == true ? "PASS" : value == false ? "FAIL" : "NOT COMPUTED",
            detail: detail,
            tone: value == true ? .good : value == false ? .bad : .neutral
        )
    }
}

private struct VettingCheck {
    let title: String
    let status: String
    let detail: String
    let tone: ReportTone
}

// MARK: - Native PDF drawing system

private enum ReportTone { case good, warning, bad, neutral }

private enum Palette {
    static let navy = NSColor(calibratedRed: 0.035, green: 0.090, blue: 0.145, alpha: 1)
    static let slate = NSColor(calibratedRed: 0.29, green: 0.35, blue: 0.41, alpha: 1)
    static let pale = NSColor(calibratedRed: 0.955, green: 0.968, blue: 0.977, alpha: 1)
    static let rule = NSColor(calibratedRed: 0.82, green: 0.85, blue: 0.88, alpha: 1)
    static let cyan = NSColor(calibratedRed: 0.00, green: 0.48, blue: 0.72, alpha: 1)
    static let green = NSColor(calibratedRed: 0.06, green: 0.49, blue: 0.34, alpha: 1)
    static let amber = NSColor(calibratedRed: 0.78, green: 0.47, blue: 0.02, alpha: 1)
    static let red = NSColor(calibratedRed: 0.73, green: 0.17, blue: 0.18, alpha: 1)
    static let white = NSColor.white

    static func color(_ tone: ReportTone) -> NSColor {
        switch tone {
        case .good: green
        case .warning: amber
        case .bad: red
        case .neutral: cyan
        }
    }
}

private final class PDFCanvas {
    static let pageRect = CGRect(x: 0, y: 0, width: 842, height: 595)

    enum TextStyle {
        case display, pageTitle, section, metric, body, small, caption, eyebrow, mono, table

        var font: NSFont {
            switch self {
            case .display: fontNamed("Avenir Next Demi Bold", 25, weight: .bold)
            case .pageTitle: fontNamed("Avenir Next Demi Bold", 17, weight: .semibold)
            case .section: fontNamed("Avenir Next Demi Bold", 10.5, weight: .semibold)
            case .metric: NSFont.monospacedDigitSystemFont(ofSize: 14, weight: .semibold)
            case .body: fontNamed("Avenir Next", 10, weight: .regular)
            case .small: fontNamed("Avenir Next", 9, weight: .regular)
            case .caption: fontNamed("Avenir Next", 8, weight: .regular)
            case .eyebrow: fontNamed("Avenir Next Demi Bold", 8, weight: .semibold)
            case .mono: NSFont.monospacedSystemFont(ofSize: 7.4, weight: .regular)
            case .table: NSFont.monospacedDigitSystemFont(ofSize: 7, weight: .regular)
            }
        }
    }

    struct TableColumn {
        let title: String
        let width: CGFloat
        let alignment: NSTextAlignment
        init(_ title: String, _ width: CGFloat, _ alignment: NSTextAlignment) {
            self.title = title; self.width = width; self.alignment = alignment
        }
    }

    let context: CGContext
    private let runID: String
    private let candidateID: String
    private let generatedAt: Date
    private var appKitContext: NSGraphicsContext?
    private var page = 0

    init(context: CGContext, runID: String, candidateID: String, generatedAt: Date) {
        self.context = context
        self.runID = runID
        self.candidateID = candidateID
        self.generatedAt = generatedAt
    }

    func beginPage(section: String, title: String) {
        page += 1
        context.beginPDFPage(nil)
        context.saveGState()
        context.translateBy(x: 0, y: Self.pageRect.height)
        context.scaleBy(x: 1, y: -1)
        let graphics = NSGraphicsContext(cgContext: context, flipped: true)
        appKitContext = graphics
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = graphics

        fill(Self.pageRect, Palette.white)
        fill(CGRect(x: 0, y: 0, width: Self.pageRect.width, height: 56), Palette.navy)
        fill(CGRect(x: 0, y: 55, width: Self.pageRect.width, height: 2), Palette.cyan)
        context.setStrokeColor(Palette.cyan.cgColor)
        context.setLineWidth(1.2)
        context.strokeEllipse(in: CGRect(x: 23, y: 17, width: 21, height: 21))
        fill(CGRect(x: 40, y: 25.5, width: 8, height: 2), Palette.cyan)

        text("AVRIL", in: CGRect(x: 54, y: 15, width: 75, height: 16), style: .section, color: Palette.white)
        text("EXOPLANET WORKSTATION", in: CGRect(x: 54, y: 31, width: 150, height: 12), style: .eyebrow, color: Palette.white.withAlphaComponent(0.62))
        text(section, in: CGRect(x: 206, y: 15, width: 210, height: 13), style: .eyebrow, color: Palette.cyan)
        text(title, in: CGRect(x: 206, y: 28, width: 470, height: 20), style: .pageTitle, color: Palette.white)
        text("PAGE \(page)", in: CGRect(x: 746, y: 22, width: 62, height: 15), style: .eyebrow, color: Palette.white, alignment: .right)

        fill(CGRect(x: 0, y: 564, width: 842, height: 31), Palette.pale)
        line(from: CGPoint(x: 34, y: 564), to: CGPoint(x: 808, y: 564), color: Palette.rule, width: 0.6)
        text("JOB \(runID)  ·  \(candidateID)", in: CGRect(x: 34, y: 573, width: 410, height: 12), style: .mono, color: Palette.slate)
        let stamp = Self.timestamp.string(from: generatedAt)
        text("GENERATED \(stamp)", in: CGRect(x: 500, y: 573, width: 308, height: 12), style: .mono, color: Palette.slate, alignment: .right)
    }

    func endPage() {
        NSGraphicsContext.restoreGraphicsState()
        appKitContext = nil
        context.restoreGState()
        context.endPDFPage()
    }

    func text(
        _ string: String,
        in rect: CGRect,
        style: TextStyle,
        color: NSColor,
        alignment: NSTextAlignment = .left,
        lineBreak: NSLineBreakMode = .byWordWrapping
    ) {
        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = alignment
        paragraph.lineBreakMode = lineBreak
        paragraph.minimumLineHeight = style == .caption ? 10 : 0
        NSAttributedString(
            string: string,
            attributes: [
                .font: style.font,
                .foregroundColor: color,
                .paragraphStyle: paragraph,
            ]
        ).draw(in: rect)
    }

    func sectionTitle(_ title: String, y: CGFloat, x: CGFloat = 34, width: CGFloat = 774) {
        fill(CGRect(x: x, y: y, width: 3, height: 15), Palette.cyan)
        text(title.uppercased(), in: CGRect(x: x + 10, y: y, width: width - 10, height: 16), style: .section, color: Palette.navy)
    }

    func metricCard(frame: CGRect, label: String, value: String, note: String) {
        rounded(frame, radius: 5, fill: Palette.pale, stroke: Palette.rule)
        fill(CGRect(x: frame.minX, y: frame.minY, width: 3, height: frame.height), Palette.cyan)
        text(label, in: frame.insetBy(dx: 12, dy: 9), style: .eyebrow, color: Palette.slate)
        text(value, in: CGRect(x: frame.minX + 12, y: frame.minY + 27, width: frame.width - 24, height: 22), style: .metric, color: Palette.navy, lineBreak: .byTruncatingTail)
        text(note, in: CGRect(x: frame.minX + 12, y: frame.maxY - 18, width: frame.width - 24, height: 11), style: .caption, color: Palette.slate)
    }

    func statusBanner(frame: CGRect, title: String, detail: String, tone: ReportTone) {
        let tint = Palette.color(tone)
        rounded(frame, radius: 5, fill: tint.withAlphaComponent(0.08), stroke: tint.withAlphaComponent(0.38))
        fill(CGRect(x: frame.minX, y: frame.minY, width: 4, height: frame.height), tint)
        text(title, in: CGRect(x: frame.minX + 14, y: frame.minY + 11, width: frame.width - 25, height: 16), style: .section, color: tint)
        text(detail, in: CGRect(x: frame.minX + 14, y: frame.minY + 31, width: frame.width - 25, height: frame.height - 39), style: .caption, color: Palette.slate)
    }

    func notice(_ message: String, frame: CGRect) {
        rounded(frame, radius: 4, fill: Palette.pale, stroke: Palette.rule)
        text(message, in: frame.insetBy(dx: 12, dy: 8), style: .caption, color: Palette.slate)
    }

    func unavailable(title: String, detail: String, frame: CGRect) {
        rounded(frame, radius: 5, fill: Palette.pale, stroke: Palette.rule)
        line(from: CGPoint(x: frame.minX + 18, y: frame.midY), to: CGPoint(x: frame.minX + 56, y: frame.midY), color: Palette.slate, width: 1.4)
        text(title, in: CGRect(x: frame.minX + 76, y: frame.minY + 35, width: frame.width - 100, height: 20), style: .section, color: Palette.navy)
        text(detail, in: CGRect(x: frame.minX + 76, y: frame.minY + 61, width: frame.width - 100, height: 48), style: .body, color: Palette.slate)
    }

    func pill(_ title: String, x: CGFloat, y: CGFloat, tone: ReportTone) {
        let tint = Palette.color(tone)
        let width = max(116, CGFloat(title.count) * 6.5 + 24)
        rounded(CGRect(x: x, y: y, width: width, height: 25), radius: 12.5, fill: tint.withAlphaComponent(0.1), stroke: tint.withAlphaComponent(0.38))
        text(title, in: CGRect(x: x + 10, y: y + 6, width: width - 20, height: 13), style: .eyebrow, color: tint, alignment: .center)
    }

    func keyValueGrid(rows: [(String, String)], frame: CGRect, columns: Int) {
        let safeColumns = max(1, columns)
        let rowCount = max(1, Int(ceil(Double(rows.count) / Double(safeColumns))))
        let cellWidth = frame.width / CGFloat(safeColumns)
        let cellHeight = frame.height / CGFloat(rowCount)
        rounded(frame, radius: 4, fill: Palette.white, stroke: Palette.rule)
        for index in rows.indices {
            let column = index % safeColumns
            let row = index / safeColumns
            let rect = CGRect(
                x: frame.minX + CGFloat(column) * cellWidth,
                y: frame.minY + CGFloat(row) * cellHeight,
                width: cellWidth,
                height: cellHeight
            )
            if column > 0 { line(from: CGPoint(x: rect.minX, y: rect.minY + 5), to: CGPoint(x: rect.minX, y: rect.maxY - 5), color: Palette.rule, width: 0.5) }
            if row > 0 { line(from: CGPoint(x: rect.minX + 6, y: rect.minY), to: CGPoint(x: rect.maxX - 6, y: rect.minY), color: Palette.rule, width: 0.5) }
            text(rows[index].0.uppercased(), in: CGRect(x: rect.minX + 9, y: rect.minY + 5, width: rect.width - 18, height: 11), style: .eyebrow, color: Palette.slate, lineBreak: .byTruncatingTail)
            text(rows[index].1, in: CGRect(x: rect.minX + 9, y: rect.minY + 18, width: rect.width - 18, height: max(12, rect.height - 22)), style: .small, color: rows[index].1 == "NOT COMPUTED" ? Palette.slate : Palette.navy, lineBreak: .byTruncatingTail)
        }
    }

    func funnel(_ rows: [(String, Int?)], frame: CGRect) {
        rounded(frame, radius: 4, fill: Palette.white, stroke: Palette.rule)
        let known = rows.compactMap(\.1)
        let maximum = max(known.max() ?? 0, 1)
        let lineHeight = frame.height / CGFloat(max(rows.count, 1))
        for (index, item) in rows.enumerated() {
            let y = frame.minY + CGFloat(index) * lineHeight
            text(item.0, in: CGRect(x: frame.minX + 10, y: y + 7, width: 168, height: 12), style: .eyebrow, color: Palette.slate)
            let track = CGRect(x: frame.minX + 182, y: y + 8, width: frame.width - 234, height: 12)
            rounded(track, radius: 2, fill: Palette.pale, stroke: nil)
            if let count = item.1 {
                let fraction = max(0.015, min(1, CGFloat(count) / CGFloat(maximum)))
                rounded(CGRect(x: track.minX, y: track.minY, width: track.width * fraction, height: track.height), radius: 2, fill: Palette.cyan.withAlphaComponent(0.78), stroke: nil)
            }
            text(item.1.map(String.init) ?? "NOT COMPUTED", in: CGRect(x: frame.maxX - 48, y: y + 6, width: 38, height: 14), style: .mono, color: Palette.navy, alignment: .right)
        }
    }

    func foldPlot(title: String, series: CandidateFoldSeries, frame: CGRect, color: NSColor, durationHours: Double, periodDays: Double) {
        rounded(frame, radius: 4, fill: Palette.white, stroke: Palette.rule)
        text(title, in: CGRect(x: frame.minX + 12, y: frame.minY + 9, width: 180, height: 14), style: .section, color: Palette.navy)
        text("\(series.returnedBinCount) bins · \(series.qualityCleanSamples) clean samples", in: CGRect(x: frame.maxX - 190, y: frame.minY + 10, width: 178, height: 13), style: .mono, color: Palette.slate, alignment: .right)

        let plot = CGRect(x: frame.minX + 46, y: frame.minY + 39, width: frame.width - 61, height: frame.height - 73)
        fill(plot, Palette.pale.withAlphaComponent(0.5))
        guard !series.bins.isEmpty else {
            text("NOT COMPUTED", in: plot, style: .section, color: Palette.slate, alignment: .center)
            return
        }
        let bins = series.bins.filter { $0.phaseCycles.isFinite && $0.medianPPM.isFinite && $0.lowerPPM.isFinite && $0.upperPPM.isFinite }
        guard !bins.isEmpty else {
            text("NOT COMPUTED", in: plot, style: .section, color: Palette.slate, alignment: .center)
            return
        }
        let minX = bins.map(\.phaseCycles).min() ?? -0.5
        let maxX = bins.map(\.phaseCycles).max() ?? 0.5
        var minY = bins.map(\.lowerPPM).min() ?? -1
        var maxY = bins.map(\.upperPPM).max() ?? 1
        if minY == maxY { minY -= 1; maxY += 1 }
        let padding = (maxY - minY) * 0.08
        minY -= padding; maxY += padding
        func px(_ value: Double) -> CGFloat { plot.minX + CGFloat((value - minX) / max(maxX - minX, 1e-12)) * plot.width }
        func py(_ value: Double) -> CGFloat { plot.maxY - CGFloat((value - minY) / max(maxY - minY, 1e-12)) * plot.height }

        if periodDays > 0 {
            let halfDuration = (durationHours / 24) / periodDays / 2
            let transit = CGRect(x: px(-halfDuration), y: plot.minY, width: max(1, px(halfDuration) - px(-halfDuration)), height: plot.height)
            fill(transit, Palette.amber.withAlphaComponent(0.08))
        }
        line(from: CGPoint(x: px(0), y: plot.minY), to: CGPoint(x: px(0), y: plot.maxY), color: Palette.amber.withAlphaComponent(0.7), width: 0.8)
        if minY <= 0, maxY >= 0 {
            line(from: CGPoint(x: plot.minX, y: py(0)), to: CGPoint(x: plot.maxX, y: py(0)), color: Palette.rule, width: 0.7)
        }

        context.setFillColor(color.withAlphaComponent(0.13).cgColor)
        context.beginPath()
        context.move(to: CGPoint(x: px(bins[0].phaseCycles), y: py(bins[0].lowerPPM)))
        for bin in bins.dropFirst() { context.addLine(to: CGPoint(x: px(bin.phaseCycles), y: py(bin.lowerPPM))) }
        for bin in bins.reversed() { context.addLine(to: CGPoint(x: px(bin.phaseCycles), y: py(bin.upperPPM))) }
        context.closePath(); context.fillPath()

        context.setStrokeColor(color.cgColor)
        context.setLineWidth(1.25)
        context.beginPath()
        context.move(to: CGPoint(x: px(bins[0].phaseCycles), y: py(bins[0].medianPPM)))
        for bin in bins.dropFirst() { context.addLine(to: CGPoint(x: px(bin.phaseCycles), y: py(bin.medianPPM))) }
        context.strokePath()

        text(String(format: "%.3f", maxY), in: CGRect(x: frame.minX + 4, y: plot.minY - 4, width: 38, height: 11), style: .mono, color: Palette.slate, alignment: .right)
        text(String(format: "%.3f", minY), in: CGRect(x: frame.minX + 4, y: plot.maxY - 8, width: 38, height: 11), style: .mono, color: Palette.slate, alignment: .right)
        text(String(format: "%.4f", minX), in: CGRect(x: plot.minX, y: plot.maxY + 6, width: 50, height: 11), style: .mono, color: Palette.slate)
        text(String(format: "%.4f", maxX), in: CGRect(x: plot.maxX - 50, y: plot.maxY + 6, width: 50, height: 11), style: .mono, color: Palette.slate, alignment: .right)
        text("phase [cycles]", in: CGRect(x: plot.midX - 45, y: plot.maxY + 6, width: 90, height: 11), style: .caption, color: Palette.slate, alignment: .center)
    }

    func tableHeader(columns: [TableColumn], x: CGFloat, y: CGFloat) {
        let width = columns.reduce(0) { $0 + $1.width }
        fill(CGRect(x: x, y: y, width: width, height: 26), Palette.navy)
        var offset = x
        for column in columns {
            text(column.title, in: CGRect(x: offset + 3, y: y + 8, width: column.width - 6, height: 11), style: .eyebrow, color: Palette.white, alignment: column.alignment, lineBreak: .byTruncatingTail)
            offset += column.width
        }
    }

    func tableRow(values: [String], columns: [TableColumn], x: CGFloat, y: CGFloat, tone: ReportTone, alternate: Bool) {
        let width = columns.reduce(0) { $0 + $1.width }
        fill(CGRect(x: x, y: y, width: width, height: 25), alternate ? Palette.pale.withAlphaComponent(0.75) : Palette.white)
        line(from: CGPoint(x: x, y: y + 25), to: CGPoint(x: x + width, y: y + 25), color: Palette.rule, width: 0.4)
        var offset = x
        for (index, column) in columns.enumerated() {
            let value = index < values.count ? values[index] : "NOT COMPUTED"
            let color = index == columns.count - 1 ? Palette.color(tone) : Palette.navy
            text(value, in: CGRect(x: offset + 3, y: y + 8, width: column.width - 6, height: 11), style: .table, color: color, alignment: column.alignment, lineBreak: .byTruncatingTail)
            offset += column.width
        }
    }

    func vettingCard(frame: CGRect, title: String, status: String, detail: String, tone: ReportTone) {
        let tint = Palette.color(tone)
        rounded(frame, radius: 4, fill: Palette.white, stroke: Palette.rule)
        fill(CGRect(x: frame.minX, y: frame.minY, width: 3, height: frame.height), tint)
        text(title.uppercased(), in: CGRect(x: frame.minX + 11, y: frame.minY + 7, width: 195, height: 12), style: .eyebrow, color: Palette.slate, lineBreak: .byTruncatingTail)
        text(status.uppercased(), in: CGRect(x: frame.maxX - 155, y: frame.minY + 7, width: 144, height: 12), style: .eyebrow, color: tint, alignment: .right, lineBreak: .byTruncatingTail)
        text(detail, in: CGRect(x: frame.minX + 11, y: frame.minY + 24, width: frame.width - 22, height: 20), style: .caption, color: Palette.navy, lineBreak: .byTruncatingTail)
    }

    private func fill(_ rect: CGRect, _ color: NSColor) {
        context.setFillColor(color.cgColor)
        context.fill(rect)
    }

    private func rounded(_ rect: CGRect, radius: CGFloat, fill: NSColor, stroke: NSColor?) {
        let path = CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
        context.addPath(path)
        context.setFillColor(fill.cgColor)
        context.fillPath()
        if let stroke {
            context.addPath(path)
            context.setStrokeColor(stroke.cgColor)
            context.setLineWidth(0.7)
            context.strokePath()
        }
    }

    private func line(from: CGPoint, to: CGPoint, color: NSColor, width: CGFloat) {
        context.setStrokeColor(color.cgColor)
        context.setLineWidth(width)
        context.move(to: from)
        context.addLine(to: to)
        context.strokePath()
    }

    private static func fontNamed(_ name: String, _ size: CGFloat, weight: NSFont.Weight) -> NSFont {
        NSFont(name: name, size: size) ?? NSFont.systemFont(ofSize: size, weight: weight)
    }

    private static let timestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss ZZZZZ"
        return formatter
    }()
}

// MARK: - Strict scientific formatting

private func optional(_ value: String?) -> String {
    guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return "NOT COMPUTED" }
    return value
}

private func value<T>(_ value: T?) -> String {
    value.map { String(describing: $0) } ?? "NOT COMPUTED"
}

private func decimal(_ value: Double?, digits: Int, suffix: String = "") -> String {
    guard let value, value.isFinite else { return "NOT COMPUTED" }
    return String(format: "%.*f%@", digits, value, suffix)
}

private func scientific(_ value: Double?) -> String {
    guard let value, value.isFinite else { return "NOT COMPUTED" }
    return String(format: "%.3e", value)
}

private func percent(_ value: Double?) -> String {
    guard let value, value.isFinite else { return "NOT COMPUTED" }
    return String(format: "%.1f%%", value * 100)
}

private func boolean(_ value: Bool?) -> String {
    guard let value else { return "NOT COMPUTED" }
    return value ? "YES" : "NO"
}

private func seconds(_ value: Double?) -> String {
    guard let value, value.isFinite else { return "NOT COMPUTED" }
    if value >= 60 { return String(format: "%.2f min", value / 60) }
    return String(format: "%.2f s", value)
}

private func list<T>(_ values: [T]) -> String {
    values.isEmpty ? "NOT COMPUTED" : values.map { String(describing: $0) }.joined(separator: ", ")
}

private func setList<T: Comparable>(_ values: Set<T>) -> String {
    values.isEmpty ? "NOT COMPUTED" : values.sorted().map { String(describing: $0) }.joined(separator: ", ")
}

private func normalized(_ value: String) -> String {
    value.replacingOccurrences(of: "_", with: " ").uppercased()
}
