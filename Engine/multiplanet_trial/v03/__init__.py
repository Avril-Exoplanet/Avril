"""Experimental v0.3 features.

This package is intentionally separate from the frozen v0.2 search.  Nothing in
here is allowed to change the published v0.2 hashes or blind-test result.
"""

from .event_search import (
    DipEvent,
    EventFirstResult,
    EventSearchConfig,
    PeriodicEventCandidate,
    detect_dip_events,
    event_first_search,
    measure_candidate_timings,
)
from .hybrid import event_result_from_dict, merge_candidate_channels
from .lightkurve_evidence import (
    FluxBranch,
    load_lightkurve_branches,
    localize_candidate_tpf,
    localize_candidates_tpf,
    localize_difference_cube,
    measure_fixed_ephemeris,
    validate_candidates_across_branches,
)
from .archive import (
    configure_lightkurve_cache,
    discover_products,
    download_inventory_resumable,
    file_integrity_record,
)
from .multibranch_search import (
    SearchBranch,
    event_first_candidates,
    fuse_branch_candidates,
    load_protected_pld_branch,
    load_mission_lightcurve_branches,
    load_spoc_independent_branches,
    load_tpf_aperture_branches,
    measure_clusters_on_all_branches,
    run_postprocessing_injection_check,
    run_cbv_precorrection_injection_check,
    run_pld_precorrection_injection_check,
    search_branch_bls,
)
from .pixel_vetting import (
    associate_centroid_with_neighbors,
    detailed_tpf_localization,
    query_tic_gaia_neighbors,
)

__all__ = [
    "DipEvent",
    "EventFirstResult",
    "EventSearchConfig",
    "PeriodicEventCandidate",
    "detect_dip_events",
    "event_first_search",
    "measure_candidate_timings",
    "event_result_from_dict",
    "merge_candidate_channels",
    "FluxBranch",
    "load_lightkurve_branches",
    "measure_fixed_ephemeris",
    "validate_candidates_across_branches",
    "localize_difference_cube",
    "localize_candidate_tpf",
    "localize_candidates_tpf",
    "configure_lightkurve_cache",
    "discover_products",
    "download_inventory_resumable",
    "file_integrity_record",
    "SearchBranch",
    "load_spoc_independent_branches",
    "load_mission_lightcurve_branches",
    "load_tpf_aperture_branches",
    "load_protected_pld_branch",
    "search_branch_bls",
    "event_first_candidates",
    "fuse_branch_candidates",
    "measure_clusters_on_all_branches",
    "run_postprocessing_injection_check",
    "run_cbv_precorrection_injection_check",
    "run_pld_precorrection_injection_check",
    "query_tic_gaia_neighbors",
    "associate_centroid_with_neighbors",
    "detailed_tpf_localization",
]
