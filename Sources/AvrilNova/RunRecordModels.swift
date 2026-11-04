import Foundation

struct RunRecordPage: Decodable {
    let items: [RunRecordSummary]
    let total: Int
}

struct RunRecordSummary: Decodable, Identifiable, Hashable {
    let id: String
    let target: String
    let status: String
    let stage: String?
    let percent: Double?
    let message: String?
    let engine: String?
    let backend: String?
    let source: String?
    let pixels: String?
    let createdAt: String?
    let updatedAt: String?
    let startedAt: String?
    let completedAt: String?
    let queueSeconds: Double?
    let computeSeconds: Double?
    let totalSeconds: Double?
    let error: String?
    let result: JobSummary?

    enum CodingKeys: String, CodingKey {
        case id, target, status, stage, percent, message, engine, backend, source, pixels, error, result
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case startedAt = "started_at"
        case completedAt = "completed_at"
        case queueSeconds = "queue_seconds"
        case computeSeconds = "compute_seconds"
        case totalSeconds = "total_seconds"
    }

    var isComplete: Bool { status == "complete" }
    var auditRows: Int { result?.candidateCount ?? 0 }
}

/// A phase fold rendered from every quality-clean FITS sample at the sealed
/// candidate ephemeris.  This is intentionally distinct from the bounded
/// light-curve preview: the bridge does not decimate or refit this evidence.
struct CandidateFoldDocument: Decodable, Sendable {
    let schemaVersion: String
    let jobID: String
    let candidateID: String
    let ephemerisLocked: Bool
    let periodDays: Double
    let epochBTJD: Double
    let durationHours: Double
    let durationDays: Double
    let phaseWindow: CandidateFoldPhaseWindow
    let sectorPartition: CandidateFoldSectorPartition
    let sectors: CandidateFoldSectorSets
    let samples: CandidateFoldSampleSummary
    let method: CandidateFoldMethod
    let folds: CandidateFoldGroups

    enum CodingKeys: String, CodingKey {
        case folds, sectors, samples, method
        case schemaVersion = "schema_version"
        case jobID = "job_id"
        case candidateID = "candidate_id"
        case ephemerisLocked = "ephemeris_locked"
        case periodDays = "period_days"
        case epochBTJD = "epoch_btjd"
        case durationHours = "duration_hours"
        case durationDays = "duration_days"
        case phaseWindow = "phase_window"
        case sectorPartition = "sector_partition"
    }
}

struct CandidateFoldPhaseWindow: Decodable, Sendable {
    let minimumCycles: Double
    let maximumCycles: Double
    let halfWidthCycles: Double
    let halfWidthDays: Double
    let halfWidthDurationMultiples: Double
    let clippedAtQuarterPeriod: Bool

    enum CodingKeys: String, CodingKey {
        case minimumCycles = "minimum_cycles"
        case maximumCycles = "maximum_cycles"
        case halfWidthCycles = "half_width_cycles"
        case halfWidthDays = "half_width_days"
        case halfWidthDurationMultiples = "half_width_duration_multiples"
        case clippedAtQuarterPeriod = "clipped_at_quarter_period"
    }
}

struct CandidateFoldSectorPartition: Decodable, Sendable {
    let discovery: [Int]
    let holdout: [Int]
    let unassignedAvailable: [Int]
    let blocks: [CandidateFoldBlock]

    enum CodingKeys: String, CodingKey {
        case discovery, holdout, blocks
        case unassignedAvailable = "unassigned_available"
    }
}

struct CandidateFoldSectorSets: Decodable, Sendable {
    let discovery: [Int]
    let holdout: [Int]
    let unassignedAvailable: [Int]

    enum CodingKeys: String, CodingKey {
        case discovery, holdout
        case unassignedAvailable = "unassigned_available"
    }
}

struct CandidateFoldBlock: Decodable, Sendable, Identifiable {
    let block: Int
    let sectors: [Int]
    let group: String
    let independentlyDiscovered: Bool

    var id: Int { block }

    enum CodingKeys: String, CodingKey {
        case block, sectors, group
        case independentlyDiscovered = "independently_discovered"
    }
}

struct CandidateFoldSampleSummary: Decodable, Sendable {
    let sourceFilesResolved: Int
    let sourceFilesUsed: Int
    let sourceFilesSkipped: Int
    let qualityCleanTotal: Int
    let windowTotal: Int

    enum CodingKeys: String, CodingKey {
        case sourceFilesResolved = "source_files_resolved"
        case sourceFilesUsed = "source_files_used"
        case sourceFilesSkipped = "source_files_skipped"
        case qualityCleanTotal = "quality_clean_total"
        case windowTotal = "window_total"
    }
}

struct CandidateFoldMethod: Decodable, Sendable {
    let timeBasis: String
    let qualityFilter: String
    let fluxSelection: String
    let normalization: String
    let phaseDefinition: String
    let phaseCoordinate: String
    let aggregation: String
    let uncertainty: String
    let decimation: String
    let ephemerisRefit: String

    enum CodingKeys: String, CodingKey {
        case normalization, aggregation, uncertainty, decimation
        case timeBasis = "time_basis"
        case qualityFilter = "quality_filter"
        case fluxSelection = "flux_selection"
        case phaseDefinition = "phase_definition"
        case phaseCoordinate = "phase_coordinate"
        case ephemerisRefit = "ephemeris_refit"
    }
}

struct CandidateFoldGroups: Decodable, Sendable {
    let discovery: CandidateFoldSeries
    let holdout: CandidateFoldSeries
}

struct CandidateFoldSeries: Decodable, Sendable {
    let declaredSectors: [Int]
    let sectors: [Int]
    let fileCount: Int
    let qualityCleanSamples: Int
    let windowSamples: Int
    let products: [String]
    let requestedBinCount: Int
    let returnedBinCount: Int
    let bins: [CandidateFoldBin]

    enum CodingKeys: String, CodingKey {
        case sectors, products, bins
        case declaredSectors = "declared_sectors"
        case fileCount = "file_count"
        case qualityCleanSamples = "quality_clean_samples"
        case windowSamples = "window_samples"
        case requestedBinCount = "requested_bin_count"
        case returnedBinCount = "returned_bin_count"
    }
}

/// The bridge deliberately emits a compact positional tuple per bin.  Decode
/// it into named scientific quantities immediately so sample count cannot be
/// mistaken for an uncertainty or flux coordinate in the UI.
struct CandidateFoldBin: Decodable, Sendable, Identifiable {
    let phaseCycles: Double
    let medianPPM: Double
    let lowerPPM: Double
    let upperPPM: Double
    let sampleCount: Int

    var id: Double { phaseCycles }

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        phaseCycles = try container.decode(Double.self)
        medianPPM = try container.decode(Double.self)
        lowerPPM = try container.decode(Double.self)
        upperPPM = try container.decode(Double.self)
        sampleCount = try container.decode(Int.self)
        guard container.isAtEnd else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "A candidate fold bin must contain exactly five values."
            )
        }
    }

    var plotRow: [Double] {
        [phaseCycles, medianPPM, lowerPPM, upperPPM, Double(sampleCount)]
    }
}

struct ExecutionLogDocument: Decodable, Sendable {
    let schemaVersion: String
    let jobID: String
    let target: String?
    let status: String?
    let progress: ExecutionProgress
    let queue: ExecutionDuration
    let compute: ExecutionDuration
    let total: ExecutionDuration
    let actions: Int
    let events: [ExecutionEvent]

    enum CodingKeys: String, CodingKey {
        case target, status, progress, queue, compute, total, actions, events
        case schemaVersion = "schema_version"
        case jobID = "job_id"
    }
}

struct ExecutionProgress: Decodable, Sendable {
    let stage: String?
    let percent: Double?
    let message: String?
    let createdAt: String?
    let updatedAt: String?
    let startedAt: String?
    let completedAt: String?

    enum CodingKeys: String, CodingKey {
        case stage, percent, message
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case startedAt = "started_at"
        case completedAt = "completed_at"
    }
}

struct ExecutionDuration: Decodable, Sendable {
    let position: Int?
    let seconds: Double?
}

struct ExecutionEvent: Decodable, Sendable, Identifiable {
    let seq: Int?
    let time: String?
    let stage: String?
    let level: String?
    let title: String?
    let message: String?
    let command: String?
    let metrics: [String: JSONValue]?

    var id: String { "\(seq ?? -1)-\(time ?? "")-\(title ?? "")" }
}

struct ScienceTimelineDocument: Decodable, Sendable {
    let schemaVersion: String
    let jobID: String
    let target: String?
    let status: String?
    let minBTJD: Double?
    let maxBTJD: Double?
    let laneCount: Int
    let eventCount: Int
    let sectors: [Int]
    let eventSourcePolicy: String
    let lanes: [ScienceTimelineLane]

    enum CodingKeys: String, CodingKey {
        case target, status, sectors, lanes
        case schemaVersion = "schema_version"
        case jobID = "job_id"
        case minBTJD = "min_btjd"
        case maxBTJD = "max_btjd"
        case laneCount = "lane_count"
        case eventCount = "event_count"
        case eventSourcePolicy = "event_source_policy"
    }
}

struct ScienceTimelineLane: Decodable, Sendable, Identifiable {
    let candidateID: String
    let rank: Int?
    let periodDays: Double?
    let epochBTJD: Double?
    let durationHours: Double?
    let priority: String?
    let disposition: String?
    let eventSource: String?
    let eventCount: Int
    let events: [ScienceTimelineEvent]

    var id: String { candidateID }

    enum CodingKeys: String, CodingKey {
        case rank, priority, disposition, events
        case candidateID = "candidate_id"
        case periodDays = "period_days"
        case epochBTJD = "epoch_btjd"
        case durationHours = "duration_hours"
        case eventSource = "event_source"
        case eventCount = "event_count"
    }
}

struct ScienceTimelineEvent: Decodable, Sendable, Identifiable {
    let raw: JSONValue

    init(from decoder: Decoder) throws { raw = try JSONValue(from: decoder) }

    var segment: Int { raw["segment"]?.intValue ?? 0 }
    var centerBTJD: Double { raw["center_btjd"]?.doubleValue ?? .nan }
    var sesLike: Double? { raw["ses_like"]?.doubleValue }
    var coverage: Double? { raw["coverage"]?.doubleValue ?? raw["transit_coverage_fraction"]?.doubleValue }
    var gapRisk: Bool? { raw["gap_risk"]?.boolValue ?? raw["critical_gap_compromise"]?.boolValue }
    var id: String { "\(segment)-\(centerBTJD)" }
}

/// Forward-compatible decoder used for the scientific evidence document.  The
/// Avril schema is intentionally additive, so the native report reads known
/// fields without throwing away future evidence.
indirect enum JSONValue: Decodable, Sendable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case number(Double)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else if let value = try? container.decode([JSONValue].self) { self = .array(value) }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value") }
    }

    subscript(_ key: String) -> JSONValue? {
        guard case .object(let object) = self else { return nil }
        return object[key]
    }

    var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }
    var arrayValue: [JSONValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }
    var stringValue: String? {
        switch self {
        case .string(let value): value
        case .number(let value): value.formatted()
        case .bool(let value): value ? "true" : "false"
        default: nil
        }
    }
    var doubleValue: Double? {
        switch self {
        case .number(let value): value
        case .string(let value): Double(value)
        default: nil
        }
    }
    var intValue: Int? { doubleValue.map(Int.init) }
    var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    func value(at path: String...) -> JSONValue? {
        path.reduce(Optional(self)) { value, key in value?[key] }
    }

    func firstValue(at paths: [[String]]) -> JSONValue? {
        for path in paths {
            let result = path.reduce(Optional(self)) { value, key in value?[key] }
            if let result {
                if case .null = result { continue }
                return result
            }
        }
        return nil
    }

    var compactDescription: String {
        switch self {
        case .string(let value): value
        case .number(let value): value.formatted(.number.precision(.fractionLength(0...6)))
        case .bool(let value): value ? "true" : "false"
        case .array(let value): value.map(\.compactDescription).joined(separator: " · ")
        case .object: "{evidence object}"
        case .null: "—"
        }
    }
}

struct ScienceResultDocument: Decodable, Sendable {
    let target: String?
    let ticID: Int?
    let segments: [Int]
    let rawHypothesisCount: Int?
    let globallyStableFamilyCount: Int?
    let deduplicatedFamilyCount: Int?
    let canonicalCandidateCount: Int?
    let releaseID: String?
    let engineVersion: String?
    let productionSignificanceBackend: String?
    let scientificStatus: String?
    let timingSeconds: [String: Double]
    let scienceAnnotation: JSONValue?
    let candidates: [ResultCandidate]
    let candidateAudit: [ResultCandidate]

    enum CodingKeys: String, CodingKey {
        case target, segments, candidates
        case ticID = "tic_id"
        case rawHypothesisCount = "raw_hypothesis_count"
        case globallyStableFamilyCount = "globally_stable_family_count"
        case deduplicatedFamilyCount = "deduplicated_family_count"
        case canonicalCandidateCount = "canonical_candidate_count"
        case releaseID = "release_id"
        case engineVersion = "engine_version"
        case productionSignificanceBackend = "production_significance_backend"
        case scientificStatus = "scientific_status"
        case timingSeconds = "timing_seconds"
        case scienceAnnotation = "science_annotation"
        case candidateAudit = "candidate_audit"
        case rankedCandidates = "ranked_candidates"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        target = try container.decodeIfPresent(String.self, forKey: .target)
        ticID = try container.decodeIfPresent(Int.self, forKey: .ticID)
        segments = try container.decodeIfPresent([Int].self, forKey: .segments) ?? []
        rawHypothesisCount = try container.decodeIfPresent(Int.self, forKey: .rawHypothesisCount)
        globallyStableFamilyCount = try container.decodeIfPresent(Int.self, forKey: .globallyStableFamilyCount)
        deduplicatedFamilyCount = try container.decodeIfPresent(Int.self, forKey: .deduplicatedFamilyCount)
        canonicalCandidateCount = try container.decodeIfPresent(Int.self, forKey: .canonicalCandidateCount)
        releaseID = try container.decodeIfPresent(String.self, forKey: .releaseID)
        engineVersion = try container.decodeIfPresent(String.self, forKey: .engineVersion)
        productionSignificanceBackend = try container.decodeIfPresent(String.self, forKey: .productionSignificanceBackend)
        scientificStatus = try container.decodeIfPresent(String.self, forKey: .scientificStatus)
        timingSeconds = try container.decodeIfPresent([String: Double].self, forKey: .timingSeconds) ?? [:]
        scienceAnnotation = try container.decodeIfPresent(JSONValue.self, forKey: .scienceAnnotation)
        let decodedAudit = try container.decodeIfPresent([ResultCandidate].self, forKey: .candidateAudit) ?? []
        if !decodedAudit.isEmpty {
            // The bounded native payload already preserves every ranked row in
            // candidate_audit.  Avoid decoding the duplicate candidates array
            // (several megabytes for large runs) a second time.
            candidateAudit = decodedAudit
            candidates = decodedAudit
        } else {
            let decodedCandidates = try container.decodeIfPresent([ResultCandidate].self, forKey: .candidates)
                ?? container.decodeIfPresent([ResultCandidate].self, forKey: .rankedCandidates)
                ?? []
            candidates = decodedCandidates
            candidateAudit = decodedCandidates
        }
    }

    var auditCandidates: [ResultCandidate] { candidateAudit.isEmpty ? candidates : candidateAudit }

    func annotationInt(_ path: String...) -> Int? {
        guard let scienceAnnotation else { return nil }
        return path.reduce(Optional(scienceAnnotation)) { value, key in value?[key] }?.intValue
    }
}

struct ResultCandidate: Decodable, Identifiable, Sendable {
    let raw: JSONValue

    init(from decoder: Decoder) throws { raw = try JSONValue(from: decoder) }

    var id: String { "\(candidateID)-\(rank ?? 0)" }
    var candidateID: String { raw["candidate_id"]?.stringValue ?? raw["proposal_id"]?.stringValue ?? "UNNAMED" }
    var rank: Int? { raw["rank"]?.intValue }
    var discoveryRank: Int? { raw["discovery_rank"]?.intValue }
    var scienceRank: Int? { raw["science_rank"]?.intValue }
    var periodDays: Double? { raw["period_days"]?.doubleValue }
    var epochBTJD: Double? { raw["epoch_btjd"]?.doubleValue }
    var durationHours: Double? { raw["duration_hours"]?.doubleValue }
    var sourceStrength: Double? { raw["source_strength"]?.doubleValue }
    var ensembleScore: Double? { raw["ensemble_score"]?.doubleValue }
    var priority: String { raw.value(at: "science_review", "priority")?.stringValue ?? "unclassified" }
    var explanation: String { raw.value(at: "science_review", "explanation")?.stringValue ?? "No automated explanation was recorded." }
    var riskFlags: [String] { raw.value(at: "science_review", "risk_flags")?.arrayValue?.compactMap(\.stringValue) ?? [] }
    var disposition: String { raw.value(at: "meaningful_candidate_gate", "disposition")?.stringValue ?? "unclassified" }
    var confirmationRequired: Bool? { raw.value(at: "science_review", "confirmation_required")?.boolValue }
    var directHarmonicAlias: Bool? { raw["direct_harmonic_alias"]?.boolValue }
    var significanceStatus: String { raw.value(at: "detection_significance", "status")?.stringValue ?? "not_computed" }
    var passesReference: Bool? { raw.value(at: "detection_significance", "passes_7p1_reference")?.boolValue }
    var referenceMES: Double? { raw.value(at: "detection_significance", "reference_mes")?.doubleValue }
    /// Red-noise/phase-null calibrated statistic.  Never substitute an event
    /// coherence score here: the two quantities have different meanings.
    var calibratedMES: Double? {
        raw.firstValue(at: [
            ["detection_significance", "null_calibrated_mes_like"],
            ["detection_significance", "mes_like"],
        ])?.doubleValue
    }
    var rawCombinedMES: Double? { raw.value(at: "detection_significance", "combined", "mes_like")?.doubleValue }
    var eventCoherenceSNR: Double? { raw.value(at: "event_coherence", "combined", "winsorized_signed_snr")?.doubleValue }
    var phaseNullFAP: Double? { raw.value(at: "detection_significance", "phase_shift_null", "local_empirical_fap")?.doubleValue }
    var lookElsewhereFAP: Double? { raw.value(at: "detection_significance", "look_elsewhere_fap_estimate")?.doubleValue }
    var depthPPM: Double? {
        raw.firstValue(at: [
            ["ephemeris_refit", "weighted_depth_ppm"],
            ["detection_significance", "combined", "weighted_depth_ppm"],
            ["detection_significance", "weighted_depth_ppm"],
            ["event_coherence", "combined", "median_depth_ppm"],
        ])?.doubleValue
    }
    var significanceEventCount: Int? { raw.value(at: "detection_significance", "event_count")?.intValue }
    var testableEventCount: Int? {
        raw.firstValue(at: [
            ["holdout_verification", "event_repeatability", "testable_events"],
            ["event_coherence", "combined", "testable_events"],
        ])?.intValue
    }
    var supportedEventCount: Int? { raw.value(at: "holdout_verification", "event_repeatability", "supported_events")?.intValue }
    var eventCount: Int? {
        if let testableEventCount, testableEventCount > 0 { return testableEventCount }
        if let significanceEventCount, significanceEventCount > 0 { return significanceEventCount }
        return testableEventCount ?? significanceEventCount
    }
    var eventSupportFraction: Double? { raw.value(at: "holdout_verification", "event_repeatability", "support_fraction")?.doubleValue }
    var sourceMethods: [String] { raw["source_methods"]?.arrayValue?.compactMap(\.stringValue) ?? [] }
    var discoveredBlocks: [Int] { raw["discovered_blocks"]?.arrayValue?.compactMap(\.intValue) ?? [] }
    var blockMeasurements: [JSONValue] { raw.value(at: "holdout_verification", "block_measurements")?.arrayValue ?? [] }
    var strongBlockCount: Int { blockMeasurements.filter { $0["outcome"]?.stringValue == "strong" }.count }
    var weakBlockCount: Int { blockMeasurements.filter { $0["outcome"]?.stringValue == "weak" }.count }
    var untestableBlockCount: Int { blockMeasurements.filter { $0["outcome"]?.stringValue == "untestable" }.count }
    var contradictingBlockCount: Int {
        blockMeasurements.filter {
            let outcome = $0["outcome"]?.stringValue
            let status = $0["status"]?.stringValue
            return outcome == "contradicting" || status == "contradicting"
        }.count
    }
    var catalogRelation: String? {
        raw.firstValue(at: [
            ["meaningful_candidate_gate", "catalog_relation"],
            ["catalog_relation"],
            ["known_match", "status"],
        ])?.stringValue
    }
    var pixelStatus: String { raw.value(at: "pixel_vetting", "aggregate_status")?.stringValue ?? "not_computed" }
    var pixelSectorsRequested: Int? { raw.value(at: "pixel_vetting", "sectors_requested")?.intValue }
    var pixelSectorsAvailable: Int? { raw.value(at: "pixel_vetting", "sectors_available")?.intValue }
    var discoverySectors: Set<Int> {
        Set(blockMeasurements.filter {
            $0["independently_discovered"]?.boolValue == true || discoveredBlocks.contains($0["block"]?.intValue ?? -1)
        }.flatMap { $0["segments"]?.arrayValue?.compactMap(\.intValue) ?? [] })
    }
    var holdoutSectors: Set<Int> {
        Set(blockMeasurements.filter {
            $0["independently_discovered"]?.boolValue != true && !discoveredBlocks.contains($0["block"]?.intValue ?? -1)
        }.flatMap { $0["segments"]?.arrayValue?.compactMap(\.intValue) ?? [] })
    }
    var gapDiagnostics: JSONValue? {
        raw.firstValue(at: [
            ["meaningful_candidate_gate", "event_gap_diagnostics"],
            ["detection_significance", "event_gap_diagnostics"],
        ])
    }
    var eventRows: [JSONValue] {
        raw.firstValue(at: [
            ["meaningful_candidate_gate", "event_gap_diagnostics", "events"],
            ["detection_significance", "event_gap_diagnostics", "events"],
            ["detection_significance", "events"],
            ["events"],
        ])?.arrayValue ?? []
    }
    var discoveryFoldBins: [[Double]] { foldBins("discovery") }
    var holdoutFoldBins: [[Double]] { foldBins("holdout") }

    private func foldBins(_ name: String) -> [[Double]] {
        raw.value(at: "fold", name, "bins")?.arrayValue?.compactMap { row in
            guard let values = row.arrayValue else { return nil }
            let numbers = values.compactMap(\.doubleValue)
            return numbers.count >= 2 ? numbers : nil
        } ?? []
    }
}
