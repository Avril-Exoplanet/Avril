import Foundation
import SwiftUI

enum WorkspaceSection: String, CaseIterable, Identifiable {
    case workbench
    case pixels
    case review
    case settings

    var id: String { rawValue }
    var title: String {
        switch self {
        case .workbench: "Workbench"
        case .pixels: "Pixel Validation"
        case .review: "Run Records"
        case .settings: "Settings"
        }
    }
    var shortTitle: String {
        switch self {
        case .workbench: "Workbench"
        case .pixels: "Pixels"
        case .review: "Records"
        case .settings: "Settings"
        }
    }
    var symbol: String {
        switch self {
        case .workbench: "scope"
        case .pixels: "square.grid.3x3.fill"
        case .review: "doc.text.magnifyingglass"
        case .settings: "slider.horizontal.3"
        }
    }
}

enum EngineState: Equatable {
    case starting
    case ready(version: String)
    case failed(String)

    var label: String {
        switch self {
        case .starting: "Starting local science engine"
        case .ready(let version): "Avril \(version) ready"
        case .failed: "Engine unavailable"
        }
    }
}

struct TargetPage: Decodable {
    let items: [TargetRecord]
    let total: Int
}

struct TargetRecord: Decodable, Identifiable, Hashable {
    let ticID: Int
    let queueRank: Int
    let discoveryLane: String
    let laneScore: Double?
    let tmag: Double?
    let teffK: Double?
    let radiusSolar: Double?
    let massSolar: Double?
    let knownPlanets: Int?
    let catalogExpectedCountLowerBound: Double?
    let catalogPMoreThanKnown: Double?
    let gapPercentile: Double?
    let gapScore: Double?
    let detectabilityPercentile: Double?
    let sectorCount: Int?
    let sectors: [Int]?
    let recentSectorCount: Int?
    let sectorSpan: Int?
    let multiEpochCoverage: Bool?
    let modelApplicability: String?
    let searchRecipe: String?

    var id: Int { ticID }
    var title: String { "TIC \(ticID)" }
    var priority: Double { laneScore ?? 0 }

    enum CodingKeys: String, CodingKey {
        case ticID = "tic_id"
        case queueRank = "queue_rank"
        case discoveryLane = "discovery_lane"
        case laneScore = "lane_score"
        case tmag
        case teffK = "teff_k"
        case radiusSolar = "radius_solar"
        case massSolar = "mass_solar"
        case knownPlanets = "known_planets"
        case catalogExpectedCountLowerBound = "catalog_expected_count_lower_bound"
        case catalogPMoreThanKnown = "catalog_p_more_than_known"
        case gapPercentile = "gap_percentile"
        case gapScore = "gap_score"
        case detectabilityPercentile = "detectability_percentile"
        case sectorCount = "sector_count"
        case sectors
        case recentSectorCount = "recent_sector_count"
        case sectorSpan = "sector_span"
        case multiEpochCoverage = "multi_epoch_coverage"
        case modelApplicability = "model_applicability"
        case searchRecipe = "search_recipe"
    }
}

struct JobEvent: Decodable, Hashable {
    let seq: Int?
    let at: String?
    let stage: String?
    let level: String?
    let title: String?
    let message: String?
    let metrics: JobEventMetrics?
}

/// Structured counters emitted by the Avril V3 worker.  Every field is
/// optional because individual pipeline events only publish the measurements
/// they own; unknown future metrics remain forward-compatible.
struct JobEventMetrics: Decodable, Hashable {
    let filesComplete: Int?
    let filesTotal: Int?
    let failures: Int?
    let filesPerSecond: Double?
    let block: Int?
    let rawHypotheses: Int?
    let voteGroupsPerSeed: Int?
    let voteSeedCount: Int?
    let seedIndex: Int?
    let seedCount: Int?
    let familiesRemeasured: Int?
    let rankingPopulation: Int?
    let promotedFamilies: Int?
    let initialStableFamilies: Int?
    let refitEligibleFamilies: Int?
    let estimatedExactPeriodTrials: Int?
    let familiesCompleted: Int?
    let familiesTotal: Int?
    let familiesRescued: Int?
    let stableFamilies: Int?
    let families: Int?
    let auditOnlyFamilies: Int?
    let directionPassed: Int?
    let deduplicatedFamilies: Int?
    let deduplicatedCandidates: Int?
    let allCandidates: Int?
    let eligible: Int?
    let shortCircuitedButRetained: Int?
    let detailedEventMeasurementsReused: Int?
    let eligibleBeforeDirectAliasDeferral: Int?
    let firstPass: Int?
    let deferredDirectAliases: Int?
    let phaseTrialsEach: Int?
    let candidates: Int?
    let candidatesCompleted: Int?
    let candidatesTotal: Int?
    let checksEach: Int?
    let workers: Int?
    let candidate: Int?
    let total: Int?
    let targetPixelSegments: Int?
    let manualReviewRequired: Int?
    let lowPriorityOrFalseAlarmRisk: Int?
    let passes7p1Reference: Int?
    let meaningfulCandidateFunnel: MeaningfulCandidateFunnel?

    enum CodingKeys: String, CodingKey {
        case failures, block, families, candidates, workers, candidate, total, eligible
        case filesComplete = "files_complete"
        case filesTotal = "files_total"
        case filesPerSecond = "files_per_second"
        case rawHypotheses = "raw_hypotheses"
        case voteGroupsPerSeed = "vote_groups_per_seed"
        case voteSeedCount = "vote_seed_count"
        case seedIndex = "seed_index"
        case seedCount = "seed_count"
        case familiesRemeasured = "families_remeasured"
        case rankingPopulation = "ranking_population"
        case promotedFamilies = "promoted_families"
        case initialStableFamilies = "initial_stable_families"
        case refitEligibleFamilies = "refit_eligible_families"
        case estimatedExactPeriodTrials = "estimated_exact_period_trials"
        case familiesCompleted = "families_completed"
        case familiesTotal = "families_total"
        case familiesRescued = "families_rescued"
        case stableFamilies = "stable_families"
        case auditOnlyFamilies = "audit_only_families"
        case directionPassed = "direction_passed"
        case deduplicatedFamilies = "deduplicated_families"
        case deduplicatedCandidates = "deduplicated_candidates"
        case allCandidates = "all_candidates"
        case shortCircuitedButRetained = "short_circuited_but_retained"
        case detailedEventMeasurementsReused = "detailed_event_measurements_reused"
        case eligibleBeforeDirectAliasDeferral = "eligible_before_direct_alias_deferral"
        case firstPass = "first_pass"
        case deferredDirectAliases = "deferred_direct_aliases"
        case phaseTrialsEach = "phase_trials_each"
        case candidatesCompleted = "candidates_completed"
        case candidatesTotal = "candidates_total"
        case checksEach = "checks_each"
        case targetPixelSegments = "target_pixel_segments"
        case manualReviewRequired = "manual_review_required"
        case lowPriorityOrFalseAlarmRisk = "low_priority_or_false_alarm_risk"
        case passes7p1Reference = "passes_7p1_reference"
        case meaningfulCandidateFunnel = "meaningful_candidate_funnel"
    }
}

struct MeaningfulCandidateFunnel: Decodable, Hashable {
    let inputCandidates: Int?
    let meaningfulCandidates: Int?
    let shortCircuitedCandidates: Int?
    let dispositionCounts: CandidateDispositionCounts?

    enum CodingKeys: String, CodingKey {
        case inputCandidates = "input_candidates"
        case meaningfulCandidates = "meaningful_candidates"
        case shortCircuitedCandidates = "short_circuited_candidates"
        case dispositionCounts = "disposition_counts"
    }
}

struct CandidateDispositionCounts: Decodable, Hashable {
    let meaningfulUnexplained: Int?
    let gapDominated: Int?
    let eventIncoherent: Int?

    enum CodingKeys: String, CodingKey {
        case meaningfulUnexplained = "meaningful_unexplained_candidate"
        case gapDominated = "gap_adjacent_event_dominated"
        case eventIncoherent = "event_incoherent"
    }
}

struct JobSummary: Decodable, Hashable {
    let target: String?
    let sourceKind: String?
    let candidateCount: Int?
    let runtimeSeconds: Double?
    let resultURL: String?
    let auditURL: String?
    let backend: String?

    enum CodingKeys: String, CodingKey {
        case target, backend
        case sourceKind = "source_kind"
        case candidateCount = "candidate_count"
        case runtimeSeconds = "runtime_seconds"
        case resultURL = "result_url"
        case auditURL = "audit_url"
    }
}

struct JobDTO: Decodable {
    let id: String
    let target: String
    let status: String
    let stage: String?
    let percent: Double?
    let message: String?
    let controlToken: String?
    let events: [JobEvent]?
    let result: JobSummary?

    enum CodingKeys: String, CodingKey {
        case id, target, status, stage, percent, message, events, result
        case controlToken = "control_token"
    }
}

enum ScienceTaskState: String, Codable {
    case queued
    case running
    case complete
    case cancelled
    case failed

    var tint: Color {
        switch self {
        case .queued: NovaTheme.inkTertiary
        case .running: NovaTheme.accent
        case .complete: NovaTheme.green
        case .cancelled: NovaTheme.amber
        case .failed: NovaTheme.red
        }
    }
}

struct ScienceTask: Identifiable, Hashable {
    let id: UUID
    let target: TargetRecord
    var state: ScienceTaskState
    var progress: Double
    var stage: String
    var message: String
    var jobID: String?
    var controlToken: String?
    var candidateCount: Int?
    var runtimeSeconds: Double?
    var events: [JobEvent]
    var createdAt: Date
}

struct LightCurvePoint: Identifiable, Hashable {
    let id = UUID()
    let time: Double
    let flux: Double
    let sector: Int
}

struct LightCurveSeriesDTO: Decodable {
    let name: String
    let sector: Int
    let points: [[Double]]
}

struct LightCurveGap: Decodable, Hashable {
    let start: Double
    let end: Double
    let durationDays: Double?
    let durationHours: Double?
    let label: String?
    let qualityBoundary: Bool?

    enum CodingKeys: String, CodingKey {
        case start, end, label
        case durationDays = "duration_days"
        case durationHours = "duration_hours"
        case qualityBoundary = "quality_boundary"
    }
}

struct LightCurvePayload: Decodable {
    let status: String
    let target: String?
    let product: String?
    let series: [LightCurveSeriesDTO]
    let sectors: [Int]
    let gaps: [LightCurveGap]
    let validSamples: Int?
    let cadenceMedianDays: Double?
    let timeLabel: String?
    let baselineDays: Double?
    let robustScatterPPM: Double?

    enum CodingKeys: String, CodingKey {
        case status, target, product, series, sectors, gaps
        case validSamples = "valid_samples"
        case cadenceMedianDays = "cadence_median_days"
        case timeLabel = "time_label"
        case baselineDays = "baseline_days"
        case robustScatterPPM = "robust_scatter_ppm"
    }

    var flattenedPoints: [LightCurvePoint] {
        series.flatMap { item in
            item.points.compactMap { pair in
                guard pair.count >= 2 else { return nil }
                return LightCurvePoint(time: pair[0], flux: pair[1], sector: item.sector)
            }
        }
    }
}
