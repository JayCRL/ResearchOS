# The ten invariants: enforcement points and tests

ResearchOS's central claim is that research direction and scientific facts are **structured, traceable
state** rather than conversation. These ten invariants are what that claim reduces to in code. Each one
is enforced by a mechanism (not a convention) and pinned by a test.

| # | Invariant | Enforcement point | Test |
|---|---|---|---|
| 1 | Agents cannot modify raw evidence | `EvidenceRegistry.update` refuses `RAW`/immutable records; `CEILING_BY_PRINCIPAL` stops the analysis agent writing raw observations; `ResourceClass.RAW_EVIDENCE` guard | `test_agent_cannot_modify_raw_evidence`, `test_resource_guards_are_principal_independent` |
| 2 | A local task cannot silently change the core question | `StateManager.save` hashes the guarded roots before/after every write; `Task.may_request_core_change()`; `TaskManager.guard_core_change_request` | `test_debug_task_cannot_promote_to_core` |
| 3 | The writer cannot create new numbers | `paper/grounding.py::verify_numbers` re-extracts every numeral and matches it against the `NumberRef` pool built from analysis artifacts; violations block compilation | `test_writer_cannot_invent_numbers`, `test_compiler_grounding_catches_a_tampered_paper` |
| 4 | Every claim has evidence | `Claim` model validator (schema) **and** `ClaimLifecycle.obligations` (lifecycle: evidence, level, analysis artifact, no blocking conflict) | `test_claim_requires_evidence` |
| 5 | Every literature claim has a source | `LiteratureClaim` validator requires page/section/quote, and requires `fulltext_verified` for `MECHANISM` claims | `test_literature_claim_requires_source` |
| 6 | `UNKNOWN` never becomes `FALSE` | `PriorArtMatrixBuilder.upgrade_unknown` requires `explicit_absence_evidence`; rendering keeps `?` and prints answered/total | `test_unknown_is_not_false` |
| 7 | External skills cannot mutate research state | `PolicyEngine.external_skill` grants only `propose.*` and refuses to add real authority; `SkillCard` rejects declared writes to protected paths | `test_external_skill_cannot_mutate_state` |
| 8 | A skill upgrade cannot break existing benchmarks | `BenchmarkRun.evaluate_regression` compares **task-level** pass sets, not just the aggregate; `SkillLifecycle.activate` refuses a failed regression | `test_skill_upgrade_requires_regression_pass` |
| 9 | Every experiment artifact traces to a source | `ArtifactRef` stores a SHA-256; `verify_artifact` re-hashes at audit time; a mismatch becomes a blocking conflict | `test_artifact_traces_to_source` |
| 10 | Rejected claims never vanish | `Claim` tombstone fields + `EntityStore.delete` refuses research kinds (invariant 10) | `test_rejected_claim_is_retained` |

## Additional enforced invariants

| Guarantee | Mechanism | Test |
|---|---|---|
| `HYPOTHESIS → SUPPORTED` is refused | `LEGAL_TRANSITIONS` graph | `test_claim_requires_evidence` |
| The event log is tamper-evident | SHA-256 hash chain (`event_hash = sha256(prev_hash ‖ canonical_json)`) | `test_event_log_detects_tampering` |
| Concurrent writes cannot silently drop a revision | `expected_revision` optimistic concurrency | `test_concurrent_state_write_is_refused` |
| A stale direction change cannot be applied to a changed project | STR stores the guarded-state hash and goes `STALE` | `test_stale_transition_is_refused_and_marked` |
| `CANDIDATE_GAP ≠ VERIFIED_NOVELTY` | `NoveltyAudit` validator + `GapRecord` validator | `test_novelty_verdict_requires_coverage_for_a_no_match_claim` |
| A paper may not cite a retired claim | `verify_claims` → `REJECTED_CLAIM_CITED` | `test_rejected_claim_is_retained` |
| Design flags, not prose, set the evidence ceiling | `assess_experiment_level_from_design` is a pure function of `ExperimentDesign` | `test_design_flags_determine_the_evidence_ceiling` |
| No LLM on the truth path | import-graph test over `models/kernel/claims/evidence/analysis/literature/importer` | `test_no_llm_imports_on_the_truth_path` |

## How to read a refusal

Refusals are the product working. A few real examples:

```
$ researchos task transition request --path core_question ...      # no --task
error: a core-state transition request must be attached to a task; free-floating direction
changes are exactly the drift the kernel prevents

$ researchos claim transition clm_… SUPPORTED                       # from HYPOTHESIS
HYPOTHESIS -> SUPPORTED is not a legal claim transition.
From HYPOTHESIS you may go to ['REJECTED', 'SUPERSEDED', 'TESTED', 'WEAKENED'].

$ researchos paper compile                                          # after a tampered draft
compilation blocked by 1 grounding violation(s)
  UNGROUNDED_NUMBER  numeral '31.7' in this sentence has no analysis artifact behind it
```

Each message names the missing capability, the missing obligation, or the missing artifact — because a
guard that only says "denied" teaches the researcher nothing.
