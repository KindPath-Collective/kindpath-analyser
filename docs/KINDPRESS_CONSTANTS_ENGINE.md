# KindPress Constants Engine (k Evolution + RMOE)

## Purpose

The KindPress constants engine builds and evolves the shared constants baseline (`k`) from real workspace signals.
It scans environment data, discovers stable repeated constants, scores them with an evolutionary RMOE metric, and
emits versioned snapshots that support:

- better semantic compression (`k + Δ`)
- explicit drift detection
- community error-correcting refinement loops
- deterministic constants-key component derivation

## Why This Matters for Social Work

This translates directly to social-work practice:

1. Shared constants (`k`):
   - common care framework language
   - safety protocol terms
   - intake/assessment field anchors
   - team communication laws

2. Individual delta (`Δ`):
   - each person/session's irreducible lived variation
   - contextual nuance that must never be erased

3. Error-correcting loop:
   - constants are re-scored as new evidence arrives
   - stale assumptions are surfaced rather than silently reused
   - disagreement becomes observable `k` drift, not interpersonal blame

## RMOE Scoring

Each candidate constant is scored with a bounded 0-1 RMOE-like score using:

- support: how broadly it appears across scanned files
- recurrence: how frequently it appears relative to max token frequency
- domain coverage: how many domains/repositories it spans
- evolutionary stability: continuity vs volatility from prior snapshot

This is intentionally evolutionary: constants can strengthen, weaken, or drop out over time.

## Security Note

`constants_key` is a deterministic key component, not a standalone secret.

Use it as one factor in multi-factor derivation:

- constants key (shared baseline)
- individual delta key (record-specific)
- external secret/salt held outside repo

Do not rely on publicly discoverable environment constants alone for confidentiality.

## Output Artifacts

Runner writes:

- `kindpress/constants/constants.latest.json`
- `kindpress/constants/constants.<UTC>.json`
- `kindpress/constants/constants.history.jsonl`

Each snapshot includes:

- roots scanned
- total files/tokens
- constants list with RMOE/confidence
- constants key fingerprint
- key derivation parameters
- community error-correction laws used

## Run

One-shot:

```bash
python -m kindpress.constants_runner
```

Continuous:

```bash
python -m kindpress.constants_runner --watch --interval-seconds 900
```

Custom roots:

```bash
python -m kindpress.constants_runner --roots "/Users/sam/dev/KindPath-Collective:/Users/sam/kindai"
```

## Community Error-Correction Laws Encoded

- add_never_delete_history
- k_alignment_before_decode
- uncertainty_must_be_routed
- stale_requires_recompute
- cross_domain_signal_preserved
