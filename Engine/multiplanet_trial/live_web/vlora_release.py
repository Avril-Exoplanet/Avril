"""Single source of truth for the Avril V3.0 scientific release."""

from __future__ import annotations


DISPLAY_NAME = "Avril"
RELEASE_ID = "avril-v3.0"
SCHEMA_VERSION = "avril-v3.0-meaningful-candidate-funnel"

MAC_SEARCH_BACKEND = "complete-grid-native-cpu-vectorized"
CUDA_SEARCH_BACKEND = "complete-grid-exact-cuda"
CUDA_REFIT_BACKEND = "complete-deterministic-global-period-grid-cuda"
SIGNIFICANCE_BACKEND = "exact-gated-cuda-or-compiled-cpu"

VOTE_ROUNDS = 5
VOTE_SEEDS = 5
PHASE_NULL_TRIALS = 4096
MIN_SEED_SUPPORT = 3
MIN_SUPPORTED_EVENTS_FOR_AUTOMATED_HIGH_PRIORITY = 4

# V3 has a separate cache namespace so earlier or experimental Numba objects
# can never be loaded as V3 kernels after an archive move or editable install.
NUMBA_CACHE_TAG = "avril-v3_0"
