"""
KindPath :: KindPress

Whole-data capture, semantic compression, and continuous recontextualisation
for the KindPath meaning layer.

---

## The Core Problem KindPress Solves

Data is currently captured in fragments. A therapy session produces a summary.
A conversation produces a conclusion. A recording produces a chart position.
The rest — the pauses, the tonal variance, the breath patterns, the spectral
divergence in the final quarter — is discarded as "waste".

This is not merely inefficient. The discarded data often contains the most
diagnostic signal. What a producer chose *not* to correct tells you more about
the emotional state of the session than what remained. The stories people do not
tell define them as much as the stories they do.

KindPress is the mechanism for giving people informed consent over the storage
and decompression of that extra energy — keeping the whole record, compressed
efficiently enough to keep forever, and recontextualising it consistently as new
understanding emerges.

---

## The k/Δ Model

Any record in the seedbank can be expressed as:

    full picture = k (shared constants) + Δ (per-record deviation)

where k is the versioned tag/constant baseline and Δ is the fingerprint —
what makes this record distinct from what the shared model already knows.

This is not bit-layer compression. It is meaning-layer compression.
Both sides of a transmission must hold the same k-version for the Δ
to decompress cleanly. k-version mismatch is detectable and quantifiable —
the drift IS the miscommunication, made visible.

---

## For Counseling and Narrative Therapy

Every therapeutic conversation generates waste data in the extractive model:
the exact words chosen, the pauses before difficult answers, the tonal quality
when a topic is avoided, the Q4 divergence from the opening frame.

KindPress stores the Δ permanently. The k layer (shared context, theoretical
framework, treatment model) can be updated without re-processing the raw record.
Recontextualisation becomes a k-update: the session is re-read under the new
theoretical frame without the client needing to re-attend.

Progress is not measured by what the person reports but by what their Δ is
doing — whether it is increasing (diversifying, becoming more distinct from the
population baseline) or converging (potentially indicating suppression).

For conflict and miscommunication: k-version mismatch IS the conflict. Two
parties disagreeing are often operating from different k-versions. k-alignment
checking makes this visible and transforms the disagreement from "you are wrong"
to "we have different baseline assumptions — let's synchronise k first."

---

## The Reasoner and Validator

The reasoner layer adds an analytical loop:
- Survey the Δ distribution across records sharing a tag
- Ask whether k is the most coherent explanation of what records share
- Propose calibration when systematic bias is detected

The validator uses both layers to gate tag revisions:
- Before any tag definition change propagates, apply it to a sample
- Measure HMoE (information density) on compressed vs decompressed representations
- Find the optimal chunk size (evolutionary HMoE maximum)
- Only proceed if the revision improves signal quality at the optimum

---

Import paths:
    from kindpress.press import encode, decode, KindPressPacket, k_alignment_check
    from kindpress.reason import analyse_delta_distribution, k_calibration_score
    from kindpress.validate import validate_tag_revision, TagRevisionReport

See kindpath-canon/FREQUENCY_FIELD_ARCHITECTURE.md for the full theoretical
framework including KindPress as therapeutic instrument.
See kindpath-canon/CONSCIOUS_CONTEXTUALISATION.md for the earlier k/Δ foundation.
"""
