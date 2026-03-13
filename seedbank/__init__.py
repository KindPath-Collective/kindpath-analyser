"""
KindPath Analyser :: Seedbank

The elder's memory. An authenticated archive of creative work with full
forensic profiles. Every deposit is a gift to the commons — technique
and analysis made freely available so anyone can learn exactly how
things were made and what they do to a listener.

No database required. Pure JSON, portable, human readable.

NOTE: Import directly from submodules to avoid name shadowing:
    from seedbank.deposit import deposit, SeedbankRecord, SeedbankReading
    from seedbank.query import search, get_baseline, compare, ...
    from seedbank.index import rebuild_index, get_stats
    from seedbank.tags_registry import (
        define_tag, revise_tag, deprecate_tag,
        flag_stale_records, get_stale_records,
        list_tags, get_tag, seed_default_tags,
    )
    from seedbank.recompute import (
        add_reading, confluent_reading, effective_n,
        corpus_effective_n_distribution,
        recompute_stale_with_fn,
        reading_history, residue_arc, corpus_residue_drift,
    )
    from seedbank.fork_log import (
        log_deposit, log_tag_revision, log_tag_propagation,
        log_record_fork, log_record_confluence,
        get_fork_events, get_k_revision_history,
        get_universe_count, get_log_stats,
        export_doctrinal_shifts,
    )

The tags_registry implements the Living Constants Principle:
tags are versioned, revisions propagate retroactively, and the delta
between old and new residue values is itself a signal.

The recompute module implements the Fork-and-Retain Principle and Combinatorial Expansion:
recomputation never overwrites — it forks. The original reading (birth universe)
is preserved alongside every subsequent reading. Each reading is valid within
its own k-universe. The baseline_version is the time coordinate in the multiverse.
confluent_reading() enables lineage confluence across distinct k-universes.
effective_n() computes the active exponent n in Φ = km^n for any record —
growing with each fork and cross as the archive enters the non-linear living regime.

See: kindpath-canon/CONSCIOUS_CONTEXTUALISATION.md
"""
