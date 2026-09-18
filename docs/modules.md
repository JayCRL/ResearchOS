# Module reports

Every module is reported in the required form: **WHAT WAS BUILT · WHY · DATA MODEL · INVARIANTS ·
TESTS · KNOWN LIMITATIONS · NEXT DEPENDENCY**. Line and test counts are from the current tree
(90 source files, ~32,000 lines; 180 tests, ~3,400 lines).

---

## 1. `models/` — the data contract (15 core objects)

* **WHAT**: pydantic v2 schemas for ResearchProject, ResearchState, Task, Experiment, Evidence,
  Analysis, Claim, Decision, LiteraturePaper, LiteratureRelation, SkillCard, SkillVersion, Audit,
  Conflict, PaperArtifact, plus ~40 supporting objects (StateTransitionRequest, QueryPlan, NoveltyAudit,
  PriorArtMatrix, BenchmarkRun, ReviewItem, TimelineEvent, AuthorNote, RedTeamReport, Interpretation…).
  Identity is ULID-shaped and sortable; `canonical_json`/`sha256_json` make hashing stable.
* **WHY**: every downstream guarantee is a schema property. `extra="forbid"` means a hand-edited YAML
  file that drifts from the contract fails loudly; cross-field validators encode the obligations
  (`SUPPORTED` needs evidence, `MECHANISM` literature claims need full text, `UNKNOWN` needs evidence to
  become `FALSE`).
* **DATA MODEL**: see above; guarded roots are declared once in `models/decision.py::GUARDED_PATHS`.
* **INVARIANTS**: 3, 4, 5, 6, plus "design flags set the evidence ceiling" (`assess_experiment_level_from_design`).
* **TESTS**: `test_models_schema.py` (29).
* **LIMITATIONS**: schema validation cannot check *semantics* (that a cited evidence id exists) — that is
  the lifecycle's job. `with_updates` re-validates the whole object, so very large objects are copied.
* **NEXT**: an index (SQLite, derived) for faster queries at scale.

## 2. `kernel/` — state, permissions, provenance, history

* **WHAT**: `StateManager` (guarded writes, revisions, optimistic concurrency), `PolicyEngine`
  (capabilities per principal), `TaskManager` (bounded tasks), `TransitionManager` (STR filing,
  staleness, human approval, decision records), `ConflictLedger` (trust-ordered, never-deleting),
  `EventLog` (append-only SHA-256 hash chain), `provenance` (git/env/config hashing, artifact
  registration and re-verification), `levels` (deterministic evidence-level derivation), `paths` (the
  `.researchos/` layout), `store` (atomic YAML writes, delete refusal).
* **WHY**: this is where "the agent cannot redefine the project" and "nobody can quietly rewrite raw
  data" become true. Guards are hash comparisons and resource classes, not code-review conventions.
* **DATA MODEL**: `research_state.yaml` (source of truth) + one YAML file per record + `events.jsonl`.
* **INVARIANTS**: 1, 2, 4, 9, 10, plus staleness, concurrency and rewrite-detection.
* **TESTS**: `test_kernel_state.py` (27) and `test_invariants_10.py` (14).
* **LIMITATIONS**: the hash chain cannot detect tail truncation without an external anchor (commit
  `head_digest()`); file-based stores glob directories, which is fine at research scale (thousands) and
  not at millions.
* **NEXT**: `researchos audit log anchor` to pin the head digest into a commit or a signed tag.

## 3. `importer/` — Research Import (archaeology)

* **WHAT**: scanner (classification, hashing, ignore rules), deterministic extractors for
  markdown/LaTeX/notes/audits, Python AST (argparse surface, metrics, seeds), configs (YAML/JSON/TOML),
  tabular data (per-row metric facts with run labels), logs, BibTeX, PDF text (optional `pypdf`, honest
  `UNPARSED` otherwise), injected-git history; conflict detection (number vs number, duplicate runs,
  missing runs, failed runs, unit-scale and dispersion guards); reconstruction into experiments,
  analyses, evidence, claims, rejected claims, decisions, open questions, notes, literature and skill
  gaps; a human review queue.
* **WHY**: the first real experience is handing over months of work. The system must recover the research
  *state* — including contradictions and rejections — instead of summarising it into prose.
* **DATA MODEL**: `Fact` (IR with source + quote + rule + confidence) → reconstructed objects, every one
  carrying `source_refs` with a file, locator, digest and quote.
* **INVARIANTS**: 4 (never above HYPOTHESIS), 6, 9, 10, plus "conflicts are never silently resolved" and
  "the core question changes through an STR".
* **TESTS**: `test_import_dla.py` (19) against `examples/DLA` (a deliberately messy legacy project: stale
  audit, duplicated run row, diary/README contradiction, overclaiming draft, bibliography without full
  texts).
* **LIMITATIONS**: heuristics are conservative by design (they under-recover); PDF import depends on
  `pypdf` and never fabricates text; figures/tables inside PDFs are not parsed; re-importing the same
  source twice creates new candidates (no cross-batch dedup yet).
* **NEXT**: cross-batch dedup keyed on fact digests; an `--update` mode that supersedes rather than
  duplicates.

## 4. `evidence/` — Evidence OS

* **WHAT**: `EvidenceRegistry` (create with per-principal maturity ceilings, verify by re-hashing,
  link to claims, refuse content edits), `EvidenceGraph` (claim → analysis → experiment → artifact →
  code provenance; orphans, hash mismatches, integrity summary).
* **WHY**: evidence is the only legitimate origin of a scientific fact, so it must be immutable in
  practice and traceable in full.
* **DATA MODEL**: `Evidence` (type RAW→ANALYZED→VERIFIED→INTERPRETED→CLAIMED, level, artifacts,
  verification status, provenance).
* **INVARIANTS**: 1, 9.
* **TESTS**: covered by `test_invariants_10.py` and `test_kernel_state.py`.
* **LIMITATIONS**: verification re-hashes files on disk; remote/hosted artifacts are recorded as
  unverifiable rather than verified.
* **NEXT**: an evidence-diff command (`researchos evidence diff <a> <b>`).

## 5. `claims/` — Claim OS

* **WHAT**: `ClaimLifecycle` (legal transitions, per-rung obligations, derived evidence level persisted
  on every transition, tombstone semantics), `ClaimRegistry` (live/retired/paper-ready/blocked/overclaim
  queries), `language` (language classes, minimum evidence levels, deterministic weakening, drift
  detection).
* **WHY**: claim strength is where research software usually lies. The ladder makes each rung cost
  something, and the language table makes "we show" mechanically unavailable to a correlational result.
* **DATA MODEL**: `Claim` + append-only `ClaimHistoryEntry`; `language_strength` and `evidence_level` are
  derived, never authored.
* **INVARIANTS**: 4, 10, and "no skipped rung", "language cannot exceed evidence".
* **TESTS**: `test_invariants_10.py`, `test_literature_and_agents.py::test_language_strength_is_capped_by_evidence_level`.
* **LIMITATIONS**: calibration can over-weaken (it does not yet search for the *strongest still-legal*
  phrasing); the English language patterns are heuristic.
* **NEXT**: a level-aware weakening ladder that maximises permitted strength per level.

## 6. `literature/` — Literature OS

* **WHAT**: eight provider adapters (arXiv, Semantic Scholar, Crossref, OpenReview, GitHub, web, local
  BibTeX, cache) behind one interface; `QueryPlanner` over 11 query families; `coverage` (measured, with
  named deficits); `LiteratureGraph` (relations, union-find clusters, explainable closest-prior-work);
  `PriorArtMatrixBuilder` + markdown rendering with three-valued cells; `NoveltyAuditor` with a
  coverage-gated verdict.
* **WHY**: "did nobody do this?" is the question most often answered badly, and a single keyword search
  cannot license it.
* **DATA MODEL**: LiteraturePaper (abstract level vs fulltext verified), LiteratureClaim (locator
  required), LiteratureRelation, QueryPlan/PlannedQuery, LiteratureCoverage, NoveltyAudit,
  PriorArtMatrix, GapRecord.
* **INVARIANTS**: 5, 6, and "no novelty verdict without coverage".
* **TESTS**: `test_literature_and_agents.py` (offline providers, no network).
* **LIMITATIONS**: similarity is lexical (Jaccard) rather than semantic, by choice — explainability over
  recall; provider rate limits are the caller's problem; no full-text retrieval pipeline yet.
* **NEXT**: an embedding-based neighbour family *behind the same recorded-coverage discipline*.

## 7. `analysis/` — statistics and the two audits

* **WHAT**: `statistics` (mean/variance/SE/CI, Cohen's d/Hedges' g, Student/Welch/paired t-tests,
  Mann–Whitney U, bootstrap CI, Holm–Bonferroni, Benjamini–Hochberg, all with `None` + notes instead of
  invented numbers); `StatisticalAuditor` (recompute from artifacts, seed/duplicate/missing-run checks,
  baseline and budget matching, metric-definition consistency, contamination, multiple comparisons,
  prose-number comparison); `MechanismAuditor` (design → ladder rung, claim language → implied rung,
  overreach findings, alternative-explanation checklist).
* **WHY**: prose must never be the source of a number, and a controlled comparison must never be
  reported as a mechanism.
* **DATA MODEL**: `Analysis`/`StatResult` (n, mean, std, se, CI, effect size, test, df, p, paired,
  comparison family, warnings) + `Audit`/`Finding`.
* **INVARIANTS**: contributes to 3 (number provenance) and the mechanism-language discipline.
* **TESTS**: `test_literature_and_agents.py`, plus statistics coverage exercised through the chain helper.
* **LIMITATIONS**: no Bayesian or mixed-effects models; multiple-comparison correction is
  family-keyed and does not infer families.
* **NEXT**: effect-size-aware power analysis for experiment design.

## 8. `paper/` — compiler, grounding, style, readiness

* **WHAT**: `PaperCompiler` (deterministic section builders from approved material, a `NumberRef` pool
  built from analysis artifacts, calibration applied at sentence construction, refusals recorded),
  `grounding` (number, claim, citation and language gates), `style_audit` (14 deterministic rule groups
  incl. history mismatch), `readiness` (seven separate dimensions, no total).
* **WHY**: a paper is a compilation artifact. If the compiler cannot find an artifact behind a sentence,
  the sentence does not ship.
* **DATA MODEL**: GroundedSentence (claim/evidence/number/citation references), SectionDraft,
  PaperArtifact, GroundingReport/Violation, StyleFinding, ReadinessDimensionResult.
* **INVARIANTS**: 3, and "a retired claim cannot be cited", "language cannot exceed evidence".
* **TESTS**: `test_invariants_10.py`, `test_literature_and_agents.py`.
* **LIMITATIONS**: section prose is templated (an injected `TextProposer` can rewrite it, and any rewrite
  that changes a number is discarded); LaTeX output is not yet a first-class target.
* **NEXT**: a LaTeX emitter with the same grounding gates, and an author-voice model trained from imported
  notes (not from published prose).

## 9. `skills/` — Skill Intelligence Layer

* **WHAT**: `SkillRegistry` (register/route/install/audit, health), `SkillSandbox` (static denylist +
  protected-path detection + declared capability ceiling), `BenchmarkHarness` + a 20-task default
  catalogue across four suites, `SkillLifecycle` (legal transitions, activation gates, regression,
  rollback), `SkillDiscoveryAgent` (deterministic gap detection from audits/failures/conflicts,
  capability queries, synthesis A+B+rule→C), and `SkillEvolution` — an executable pipeline that runs a
  candidate through sandbox → benchmark → regression → ACTIVE or REJECT, plus
  `deterministic_runner` (a scoring runner that uses the harness's own gates) and `quality_from_benchmark`.
* **WHY**: skills are where a research system either improves or rots. "The code runs" is not quality, so
  quality is eight measured dimensions and activation is earned. Making the loop *executable* is what
  separates a lifecycle diagram from a lifecycle.
* **DATA MODEL**: SkillCard, SkillVersion, SkillCapability, SkillQualityMetrics, BenchmarkTask,
  TaskOutcome, BenchmarkRun, SandboxReport, SkillGap; `EvolutionResult` records every stage.
* **INVARIANTS**: 7, 8.
* **TESTS**: `test_timeline_redteam_evolution.py` (activation, regression rejection even when the
  aggregate score rises, sandbox rejection, event-log trail) + `test_invariants_10.py`.
* **LIMITATIONS**: the sandbox is static analysis only (it never executes untrusted code — by design);
  a real benchmark needs an injected runner, so no LLM-backed skill is benchmarked end to end yet;
  synthesis produces candidates whose *quality* is measured but whose scientific value is not.
* **NEXT**: per-skill published benchmark history and a `--baseline` comparison across releases.

## 13. `kernel/timeline.py` + `researchos timeline` — the research history (§20)

* **WHAT**: `TimelineView`, a read-only view that groups the append-only timeline into the phases a
  project actually passes through (observation → hypothesis → experiment → analysis → revision → claim →
  paper), explains any object via `why(ref=…)`, reconstructs a claim's full history
  (`claim_history`), and routes the four standing questions (why was the claim cancelled, why did the
  route change, why was this control added, why is this result not in the paper).
* **WHY**: the difference between a research log and a chat transcript is that the log can be *queried
  for reasons*. Import marks every recovered event `imported=True`, with real dates from diaries and git
  where they exist.
* **DATA MODEL**: `TimelineEvent` (kind, refs, actor, state revision, imported flag) + decisions,
  transitions and claim history joined at read time.
* **INVARIANTS**: rejected claims remain explainable (10); direction changes are always traceable (2).
* **TESTS**: `test_timeline_redteam_evolution.py`.
* **LIMITATIONS**: `why()` is keyword/ref based, not semantic; imported events without source dates are
  timestamped at import time and flagged rather than guessed.
* **NEXT**: a `timeline export --format latex` narrative for the paper's introduction.

## 14. `researchos redteam` — the adversarial loop (§19)

* **WHAT**: `RedTeamAgent.review` builds falsification questions, alternative explanations, single-seed
  and single-model dependency flags, under-matched baselines, contentious numbers and reviewer
  counter-readings; the report is persisted as a `RedTeamReport` under `paper/audits/` and recorded in
  the timeline.
* **WHY**: the weakest link is usually a claim nobody attacked. The loop must run *before* a release, and
  it must be impossible for it to "fix" the state it is criticising.
* **DATA MODEL**: `RedTeamReport`/`RedTeamQuestion`; `respects_state` is asserted, not promised.
* **INVARIANTS**: auditors report, they never mutate (the report refuses `respects_state=False`).
* **TESTS**: `test_timeline_redteam_evolution.py`.
* **LIMITATIONS**: the questions are deterministic templates over registered state; it does not read the
  paper text (the style auditor and grounding gates do that).
* **NEXT**: an `--unaddressed` view that tracks which objections a later experiment closed.

## 15. `importer/` timeline reconstruction

* **WHAT**: the import pipeline now emits `TimelineEvent`s for the material it recovers — notes (with
  their diary dates), experiments, results, claim candidates, rejected claims, decisions and open
  questions — all flagged `imported=True`.
* **WHY**: a researcher handing over a project wants the *history* back, not only the current state.
* **TESTS**: `test_import_dla.py::test_import_is_recorded_in_the_event_log_and_timeline`,
  `test_timeline_redteam_evolution.py::test_timeline_records_imported_material_as_imported`.

## 16. `api/` — read-only HTTP surface (§23)

* **WHAT**: a FastAPI app (`create_app(kernel)`) exposing health, state, claims (+ obligations),
  evidence (+ provenance trace), experiments, analyses, conflicts, audits, review queue, literature
  coverage/novelty/prior-art, timeline (+ summary, per-claim history, `why`), readiness, agents, skills
  and the capability table. `researchos api` serves it.
* **WHY**: v1 is CLI *and* API. But every mutation in ResearchOS carries an explicit principal and a
  capability check, and an HTTP write endpoint is the easiest place for ambient authority to reappear —
  so the surface is **read-only by design**, and a test asserts that write methods return 404/405.
* **DATA MODEL**: no new objects; serialised models plus derived views.
* **INVARIANTS**: preserves the permission model by not offering a bypass.
* **TESTS**: `test_api_and_voice.py` (22 route/permission assertions).
* **LIMITATIONS**: read-only (no task triggering over HTTP), no auth layer (bind to localhost or put it
  behind a proxy), no pagination.
* **NEXT**: authenticated, principal-carrying write endpoints that reuse the same gate — never a bypass.

## 17. `paper/voice.py` — researcher voice (§18)

* **WHAT**: `ResearcherVoice` reconstructs the real research arc from records (observation → hypothesis →
  anomaly → control → revision → claim), measures a style fingerprint from the researcher's **own notes**,
  detects a faked confirmation arc (`distortion`), and emits grounded, numeral-free narrative sentences
  that the compiler injects into the Introduction/Method/Discussion/Conclusion.
* **WHY**: the spec's requirement is not "sound more human" — it is that the narrative follow the research
  that actually happened. A draft claiming `hypothesis → confirmation` while the timeline shows an anomaly
  and a hypothesis revision is a false account of the method.
* **DATA MODEL**: `VoiceStep` (phase + the note/decision/claim ids it came from), `VoiceProfile`
  (sentence-length mean/std, first-person ratio, hedge ratio), `voice_context` for a writer or model.
* **INVARIANTS**: every narrative sentence points at a record or is not emitted; no numerals can enter the
  narrative (numbers must come from analysis artifacts).
* **TESTS**: `test_api_and_voice.py`.
* **LIMITATIONS**: the arc phases are deterministic templates over records; it does not learn phrasing
  from published papers (deliberately); a project with no notes produces no narrative sentences.
* **NEXT**: a `TextProposer` implementation that rewrites narrative sentences in the measured voice while
  the grounding gates re-verify them.

## 10. `agents/` — Agent OS

* **WHAT**: fifteen agents (`planner`, `importer`, `literature_researcher`, `experiment_designer`,
  `experiment`, `analysis`, `mechanism_auditor`, `novelty_auditor`, `claim_manager`, `paper_writer`,
  `paper_auditor`, `red_team`, `skill_discovery`, `skill_evaluator`, `skill_synthesizer`) as thin,
  capability-scoped wrappers over the modules, plus `Proposal` (agents propose; the kernel commits) and
  `agent_table` (which prints what each agent may **not** do).
* **WHY**: an agent that cannot commit a state change cannot drift the research line.
* **DATA MODEL**: none of its own — principals + proposals.
* **INVARIANTS**: 2, 7, and "no agent holds a human-only capability".
* **TESTS**: `test_literature_and_agents.py`.
* **LIMITATIONS**: agents are libraries, not loops; orchestration policy (when to run which agent) is
  deliberately left to the caller.
* **NEXT**: a task runner that executes a `Task`'s allowed action set with a recorded transcript.

## 11. `llm/` — optional model adapters

* **WHAT**: `LLMProvider` interface, `EchoProvider` (deterministic), OpenAI-compatible / Anthropic /
  Ollama providers with injectable HTTP, `LLMRouter` exposing only proposal operations and validating
  that a rewrite preserved every numeral, plus an explicit `PROHIBITED_USES` list and `refuse()`.
* **WHY**: models are useful for language and useless as sources of truth. The boundary is data, so it can
  be asserted in a test.
* **INVARIANTS**: "no LLM on the truth path" (`test_no_llm_imports_on_the_truth_path`).
* **TESTS**: `test_layering_and_cli.py`.
* **LIMITATIONS**: no prompt-caching or cost accounting; no automatic grounding re-verification for
  free-form generations beyond numeral preservation (the paper compiler re-verifies the final artifact
  anyway).
* **NEXT**: per-provider recorded-call transcripts for auditability.

## 12. `cli/` — the first-class interface

* **WHAT**: ~30 commands covering init/import/status/dashboard, tasks and transitions, experiments,
  evidence, claims, literature, novelty, prior art, audits (stats/mechanism/log), skills, paper
  (compile/audit/readiness), agents and the review queue.
* **WHY**: v1 is CLI and API; the UI comes after the state model is right.
* **INVARIANTS**: the CLI never approves on the user's behalf, and prints refusals as loudly as successes.
* **TESTS**: `test_layering_and_cli.py`.
* **LIMITATIONS**: no interactive TUI; long operations are synchronous.
* **NEXT**: a read-only FastAPI surface over the same kernel, then a dashboard.
