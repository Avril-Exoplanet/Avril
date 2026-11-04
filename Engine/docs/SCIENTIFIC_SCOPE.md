# Scientific scope

Avril V3.0 is a candidate-search and triage system. It is designed to maximize
auditable recovery of recurring dimming events while annotating common false
positive evidence.

It does not, by itself:

- validate or confirm a planet;
- produce a calibrated posterior planet probability;
- establish that the dimming source is on the target star;
- replace radial velocity, high-resolution imaging, or expert review;
- guarantee completeness outside a measured injection/recovery domain.

## Required evaluation sequence

1. regression on known systems without reading truth during execution;
2. injection/recovery in real TESS noise across period, depth, duration, event
   count, gaps, and harmonic interference;
3. negative controls including quiet stars, eclipsing binaries, rotation,
   red noise, instrument periods, phase shifts, and inverted light curves;
4. a new TIC-disjoint blind sample whose labels remain sealed until predictions
   and hashes are frozen;
5. natural large-sample runs with pixel localization for final priorities.

Previously opened blind sets are burned and may only be used as regressions.
Population-level claims need a new sealed split and should report recall by
depth/event-count strata, alerts per 1,000 targets, abstention rate, and runtime.

## Candidate terminology

- `candidate`: periodic evidence retained in the ledger;
- `high_priority_follow_up_candidate`: passed current automated review and is
  worth source localization/follow-up;
- `validated planet` or `confirmed planet`: external scientific status that
  Avril never assigns automatically.

## Discovery rank and science rank

Avril records two ranks rather than allowing post-discovery diagnostics to be
mistaken for part of the original search:

- `discovery_rank` is the immutable order produced by search, voting, exact
  refitting, event repeatability, and harmonic arbitration;
- `science_rank` is the public final order after red-noise, physical,
  reference, and available pixel review.

Science rank places high-priority follow-up candidates first, incomplete cases
requiring manual review second, and candidates with material false-alarm or
false-positive risk last. Within each tier the immutable discovery order is
preserved. Candidate count and locked period/epoch/duration never change.

V3 first retains but short-circuits known catalogue signals, event-proven
aliases, incoherent trains, and gap-dominated signals. It evaluates exact
phase-null significance only for remaining candidates that still have
a deterministic route to automatic high priority. Skipped rows remain in the
ledger with an explicit reason and are placed in manual or low-priority tiers.
This is a decision-preserving science gate, not peak truncation.
