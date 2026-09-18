# ResearchOS — Architecture & Repository Proposal

> **ResearchOS is a Research Operating System, not a chatbot and not an AI paper writer.**
> It makes research direction, evidence, claims, literature context and skills *structured state*
> that machines and humans can both audit.

Version: `0.1.0` · Status: design contract (frozen for the P0 implementation in `src/researchos`)

---

## 0. Three axioms

| Axiom | Consequence in the codebase |
|---|---|
| **Chat history is not research state.** | `research_state.yaml` + entity stores are the source of truth. No module may read a conversation to decide the research direction. `Task` carries an explicit boundary and priority; the kernel refuses silent promotion of a local task's finding to core. |
| **The LLM is not the source of scientific truth.** | Numbers, statistics, hashes, file existence, schema validation, citation metadata and artifact lineage are computed by deterministic code. LLM calls are confined to `researchos.llm` callers that *propose* language, and the paper pipeline re-verifies every numeral and citation against artifacts. |
| **A paper is a compilation artifact.** | The paper compiler's input is `verified evidence + approved claims + approved interpretations + literature evidence + author notes + decisions`. It cannot add a number, a citation, a mechanism, a novelty claim or a generalization. |

Rejected framings (explicitly out of scope, and asserted against in tests):
"automatic paper correctness", "automatic novelty guarantee", "automatic mechanism proof".
Substitutions: traceability, reproducibility, hallucination surface reduction, context-drift prevention,
literature coverage, causal/mechanistic discipline, overclaim reduction, auditable process.

---

## 1. Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        RESEARCHOS KERNEL                                 │
│  Global Research State · Permission Gate · State Transitions ·           │
│  Provenance · Task Boundaries · Claim/Evidence/Literature/Skill State ·  │
│  Append-only hash-chained Event Log · Conflict Ledger                     │
└───────────────┬──────────────────────────────────────────────────────────┘
                │  every mutation goes through the gate (no bypass)
   ┌────────────┼─────────────┬──────────────┬───────────────┐
   ▼            ▼             ▼              ▼               ▼
Literature    Evidence     Experiment      Skill          Timeline /
   OS           OS            OS            OS            Decision Log
   │            │             │              │               │
   └────────────┴─────────────┴──────────────┴───────────────┘
                              ▼
                    AGENT OS (15 agents, capability-scoped)
                              ▼
                          AUDITORS
             (Statistical · Mechanism · Novelty · AI-Style · Red Team)
                              ▼
                       CLAIM REGISTRY
                              ▼
                       PAPER COMPILER
                 (grounding verifier + style auditor)
```

### 1.1 Primary data flow

```
GLOBAL RESEARCH STATE
  → CURRENT TASK (bounded, prioritised)
    → AGENT EXECUTION (capability-gated)
      → EXPERIMENT / ANALYSIS / LITERATURE
        → EVIDENCE (raw → analyzed → verified → interpreted → claimed)
          → AUDIT (statistical / mechanism / novelty / style / red-team)
            → CLAIM UPDATE (lifecycle transition, evidence-gated)
              → RESEARCH STATE UPDATE (STR if core is touched)
                → PAPER COMPILER (compiles only approved material)
```

### 1.2 Skill evolution loop

```
RESEARCH FAILURE / AUDIT FINDING
  → SKILL GAP DETECTION (Skill Critic)
    → WEB / GITHUB / ARXIV SKILL DISCOVERY (capability-oriented queries)
      → CANDIDATE → SANDBOX (static safety + declared capability ceiling)
        → BENCHMARK (fixed task suite, regression vs incumbent)
          → SYNTHESIS (A + B + research rule → C, C is *untrusted*)
            → REGRESSION TEST → ACTIVE | REJECT → REGISTRY
```

---

## 2. Repository layout

```
ResearchOS/
├── ARCHITECTURE.md              ← this file (design contract)
├── README.md
├── pyproject.toml
├── docs/
│   ├── architecture/            per-subsystem design notes
│   ├── invariants.md            the 10 invariants → enforcement → test
│   ├── research_state.md        the on-disk contract
│   └── prior_art_review.md      adopt / modify / reject vs. Sakana, STORM, PaperQA2, ...
├── src/researchos/
│   ├── models/                  pydantic v2 schemas (15 core objects)
│   ├── kernel/                  state, store, events, provenance, permissions, transitions, conflicts
│   ├── importer/                Research Archaeology / Import pipeline
│   ├── literature/              providers, query planner, graph, prior-art, novelty, coverage
│   ├── evidence/                registry, provenance graph, verification
│   ├── claims/                  registry, lifecycle, grounding, language calibration
│   ├── analysis/                statistics, statistical audit, mechanism audit
│   ├── skills/                  registry, sandbox, benchmark, lifecycle, discovery, evolution
│   ├── paper/                   compiler, grounding, style auditor, readiness, templates
│   ├── agents/                  the 15 agents, capability-scoped
│   ├── llm/                     provider adapters (never on the truth path)
│   ├── api/                     optional read-mostly FastAPI surface
│   └── cli/                     `researchos <command>`
├── benchmarks/                  capability benchmark task definitions
├── examples/                    demo projects used in tests and docs
└── tests/                       unit · schema · transition · import · provenance · claim ·
                                 permission · literature · skill · invariant suites
```

---

## 3. On-disk contract (`.researchos/`)

```
.researchos/
├── project.yaml                 project identity, schema_version
├── state/
│   ├── research_state.yaml      ★ GLOBAL RESEARCH STATE — source of truth
│   ├── events.jsonl             ★ append-only, hash-chained audit log
│   ├── tasks/<task_id>.yaml
│   └── decisions/<decision_id>.yaml
├── claims/<claim_id>.yaml
├── experiments/<experiment_id>.yaml
├── evidence/
│   ├── <evidence_id>.yaml
│   └── raw/                     immutable artifact bytes (hashed)
├── analysis/<analysis_id>.yaml
├── literature/
│   ├── papers/<paper_id>.yaml
│   ├── graph/relations.jsonl
│   ├── search/<query_id>.yaml    query plans + raw provider responses
│   └── prior_art/<matrix_id>.yaml
├── skills/
│   ├── registry/<skill_id>.yaml
│   ├── active/  candidates/  deprecated/
│   └── benchmarks/<run_id>.yaml
├── audits/<audit_id>.yaml
├── conflicts/<conflict_id>.yaml
├── paper/
│   ├── artifacts/<paper_id>/     compiled sections + grounding maps
│   └── audits/
└── cache/                        disposable (providers, indexes, SQLite)
```

**Every file that holds state is a YAML/JSONL document, git-diffable and human-readable.**
SQLite is used only as a *derived, rebuildable* index and cache — never as the truth.

---

## 4. Kernel contracts

### 4.1 Principals and capabilities

A `Principal` is any actor that wants to mutate state. Agents are principals; humans are principals;
external skills are principals. Every kernel call takes an explicit principal — there is no ambient authority.

| Principal | Notable capabilities | Explicitly denied |
|---|---|---|
| `human` | everything, incl. `state.transition.approve`, `claim.approve`, `skill.activate` | — |
| `planner` | `state.read`, `state.write.task`, `state.transition.request` | core writes, raw data |
| `importer` | `state.read`, `state.write.task`, `claim.propose` (max `HYPOTHESIS`), `evidence.create` (max `ANALYZED`) | `claim.approve`, `evidence.verify`, raw write |
| `literature` | `literature.write`, `literature.novelty_audit` | any experiment/evidence mutation |
| `experiment_designer` | `experiment.register` | `experiment.update` of results |
| `experiment` | `experiment.run`, `experiment.update` (own experiment) | `claim.*`, raw evidence edit |
| `analysis` | `analysis.compute`, `evidence.create` (max `VERIFIED`) | `evidence.update` on RAW, raw data |
| `mechanism_auditor` | `audit.create` | any experiment/claim mutation |
| `novelty_auditor` | `audit.create` (novelty) | literature mutation |
| `claim_manager` | `claim.propose`, `state.transition.request` | `claim.approve` (human only) |
| `paper_writer` | `paper.read_approved`, `paper.compile` | evidence/claim/literature writes, new numbers |
| `paper_auditor` | `paper.style_audit`, `audit.create` | paper mutation |
| `red_team` | `audit.create` (red team) | silent state changes of any kind |
| `skill_*` | `skill.discover/propose/evaluate/install` | `skill.activate` (human), research state |
| `external_skill:<id>` | `procedure.propose`, `recommendation.propose`, `analysis.propose`, `draft.propose` | **all** direct state mutation |

Two enforcement layers, both mandatory:

1. **Capability check** — `PolicyEngine.require(principal, capability, resource)`.
2. **Resource guard** — a mutation of an append-only/immutable resource is rejected *regardless of principal*:
   `RAW_EVIDENCE`, `RAW_ARTIFACT`, `EVENT_LOG`, `PROVENANCE`, `REJECTED_CLAIM`, `FAILED_EXPERIMENT`.
   This is why "the agent had permission" can never be an excuse for destroying raw results.

### 4.2 Global Research State

Fields (persisted in `state/research_state.yaml`):

```
schema_version, project, revision, updated_at
core_question, core_claims[], secondary_questions[]
non_goals[], priorities[]
current_frontier, known_evidence[], rejected_claims[]
open_questions[], research_decisions[]
literature_state{coverage, closest_prior_work[], last_search_at}
skill_state{active[], gaps[], health}
active_task, task_queue[]
```

`revision` is a monotonically increasing integer. Writes are **optimistic-concurrency**:
every mutation may carry `expected_revision`; a mismatch raises `StateRevisionConflict`.
This makes "two agents silently overwrote the research plan" impossible.

**Guarded root fields** (change requires an approved STR): `core_question`, `core_claims`,
`non_goals`, `priorities`, `project.scope`. Everything else the kernel writes directly.

### 4.3 Task boundary (the anti-drift mechanism)

```yaml
task_id: tsk_...
objective: "explain why metric M drops on seed 3"
purpose: DEBUG | EXPLAIN | DESIGN | RUN | ANALYSE | LITERATURE | WRITE | IMPORT
parent_claim: clm_...            # may be null
parent_experiment: exp_...       # may be null
priority: PRIMARY | SECONDARY | EXPLORATORY
allowed_actions: [...]           # capability subset
forbidden_actions: [...]         # wins over allowed
stop_condition: "..."
created_from: research_state@revision
```

Rules the kernel enforces:

* A finding produced under a `DEBUG`/`EXPLORATORY` task is recorded as `TaskFinding` on the task.
  It **cannot** become a `core_claim` without an approved STR. (`SilentStateChangeError`.)
* `EXPLORATORY` and `SECONDARY` tasks cannot `state.transition.request` on `core_question`
  (only `PRIMARY` tasks and the human principal can).
* Any task that touches core state must declare `stop_condition`, else creation fails.
* Task completion never mutates research state directly; it produces a *proposal* +
  optional STR + optional claim proposals.

### 4.4 State Transition Request (STR)

```yaml
str_id, requested_by, created_at, task_id
current_state: { revision, state_hash }
proposed_state: { op: set|append|remove, path, value }[]
change_class: CORE_QUESTION | CORE_CLAIM | PRIORITY | NON_GOAL | SCOPE | OTHER
evidence_ids[], reason, affected_claims[], affected_experiments[]
risk: LOW | MEDIUM | HIGH
status: PROPOSED | APPROVED | REJECTED | WITHDRAWN | STALE
decided_by, decided_at, decision_note
```

* `current_state.state_hash` is a SHA-256 over the canonicalised guarded root fields; if the live
  state no longer matches, approval is refused and the STR becomes `STALE`.
* Approval requires the `state.transition.approve` capability, held **only** by `human`.
* Applying an approved STR appends a `Decision` record (never an in-place overwrite) and an event.
* `evidence_ids` must be non-empty unless `risk == LOW` and `reason` explains why.

### 4.5 Provenance and the event log

`Provenance` is attached to every experiment, evidence and analysis record:

```
code_commit, code_dirty, config_hash, dataset_hash[], model_hash, environment{}
random_seed[], timestamp, artifact_hash, command, host, python_version
```

Hashes are SHA-256; `canonical_json()` is sorted-key, UTF-8, no-whitespace JSON so hashes are stable.
`ArtifactRef(path, sha256, bytes, kind)` is verified by **re-reading and re-hashing** the bytes, so a
"verified" record whose file changed later is detected as `HASH_MISMATCH` → an automatic `Conflict`.

`state/events.jsonl` is append-only and hash-chained:

```
event_hash = sha256(prev_hash ‖ canonical_json(event_payload))
```

`researchos audit log verify` recomputes the chain; any edit, deletion or reordering is detected.
This is the "research history can always be replayed" guarantee.

### 4.6 Conflicts — never silently overwrite

```yaml
conflict_id, created_at, kind: VALUE | CLAIM | METRIC | DESIGN | LITERATURE
source_a: {kind, ref, value}, source_b: {kind, ref, value}
difference, trust_a, trust_b
resolution: A_WINS | B_WINS | BOTH_VALID | UNRESOLVED | MERGED
resolution_reason, resolved_by, confidence
preserved: all sources retained; loser marked superseded_by, never deleted
```

Deterministic trust ranking (higher wins; ties leave the conflict `UNRESOLVED`):

```
RAW_EXPERIMENT 6 > VERIFIED_ANALYSIS 5 > AUDIT 4 > EXPERIMENT_REPORT 3 > PAPER_PROSE 2 > AI_SUMMARY 1
```

A conflict never deletes either source. Unresolved conflicts block claim promotion to `ROBUST`.

### 4.7 Claim lifecycle (no skipping)

```
IDEA → HYPOTHESIS → TESTED → SUPPORTED → ROBUST
                 ↘          ↘          ↘
                  WEAKENED / REJECTED / SUPERSEDED (reachable from any state)
```

| Transition | Gate enforced by the kernel |
|---|---|
| `IDEA → HYPOTHESIS` | statement non-empty; scope declared |
| `HYPOTHESIS → TESTED` | ≥1 `supporting_experiment` in state `COMPLETED`/`FAILED` |
| `TESTED → SUPPORTED` | ≥1 evidence with level ≥ `L2_REPRODUCED` **and** ≥1 analysis artifact **and** no `UNRESOLVED` conflict touching it |
| `SUPPORTED → ROBUST` | ≥2 *independent* experiments (different seeds **or** datasets **or** scales), max level ≥ `L3_CONTROLLED`, ≥1 audit of kind `MECHANISM`/`STATISTICAL`, no `UNRESOLVED` conflict |
| `* → REJECTED` | reason + ≥1 evidence id; the claim is tombstoned (`rejected_at`, `rejection_reason`) and stays queryable forever |
| `* → SUPERSEDED` | `superseded_by` must point to an existing claim |
| **anything → `SUPPORTED` from `HYPOTHESIS`** | **rejected** (the classic overclaim jump) |

Claim fields: `claim_id, statement, status, scope, evidence_ids[], literature_ids[],
evidence_level, assumptions[], limitations[], supporting_experiments[], contradicting_experiments[],
language_strength, superseded_by, history[]`.

### 4.8 Evidence levels and language strength

| Level | Meaning | Maximum permitted language |
|---|---|---|
| `L0_IDEA` | speculation | "we hypothesise / we speculate" |
| `L1_OBSERVATION` | single observation, no control | "we observe that" |
| `L2_REPRODUCED` | reproduced across seeds/runs | "we find that" |
| `L3_CONTROLLED` | matched control/baseline | "we show that X is associated with Y" |
| `L4_INTERVENTION` | intervention changes the variable | "X causes Y" (causal verb permitted) |
| `L5_NECESSITY_OR_SUFFICIENCY` | removal/forcing isolates a component | "X is necessary/sufficient for Y" |
| `L6_CROSS_SETTING_REPLICATION` | ≥2 settings, e.g. model **and** scale | "generalises across settings" |

`evidence_level` is **derived deterministically** from the experiment's declared design
(`assess_experiment_level`), clamped to the design's ceiling — prose can never raise it.
`claims/language.py::calibrate(text, level)` detects and neutralises stronger-than-supported verbs.

### 4.9 Evidence OS

Evidence types: `RAW → ANALYZED → VERIFIED → INTERPRETED → CLAIMED`.
`RAW` is append-only and immutable. `VERIFIED` requires re-hash success, a verifier principal,
a method, and a timestamp. The evidence graph is
`Claim → Analysis → Experiment → Raw artifact → Code/Config/Environment`.

### 4.10 Literature OS

Provider adapters (nothing hard-wired): `arxiv`, `semantic_scholar`, `openreview`, `crossref`,
`github`, `web`, `local_bib`, `cache`. Each implements
`search(query) -> list[ProviderRecord]`, `fetch(id) -> ProviderRecord`, `expand(paper) -> citations|references`.

`QueryPlanner` turns a research question / mechanism / claim into an explicit, recorded query set:

```
exact_terminology, synonyms, historical_terminology,
mechanistic_equivalents, functional_equivalents,
neighboring_communities, recent_terminology,
citation_expansion, author_venue_search
```

Coverage is *measured and reported*, never assumed: `LiteratureCoverage` counts queries executed per
family, providers exercised, papers screened, and abstentions. A `NoveltyAudit` may only return
`NO_MATCH_FOUND_IN_SEARCHED_COVERAGE` when coverage thresholds are met; otherwise the verdict is forced
to `NOVELTY_UNCERTAIN`. Claim-level gap taxonomy:
`KNOWN | KNOWN_DIFFERENCE | OPEN_QUESTION | CANDIDATE_GAP | VERIFIED_NOVELTY`
with the hard rule **`CANDIDATE_GAP ≠ VERIFIED_NOVELTY`** (encoded as a guard, not a comment).

Prior-art matrix cells are `TRUE | FALSE | UNKNOWN`, each carrying `paper_id, page, section, confidence,
quote?`. **`UNKNOWN` is never auto-converted to `FALSE`** — matrix rendering keeps three-valued logic
and coverage is reported as `answered / total`.

### 4.11 Auditors

* **Statistical audit** — recomputes `n, mean, std, SE, CI, effect size, test statistic, p-value` from
  artifacts; checks seed consistency, baseline matching, training-budget matching, metric-definition
  consistency, metric/start-point contamination, missing/failed/duplicate runs, and multiple comparisons
  (Holm–Bonferroni + Benjamini–Hochberg). Numerals found in prose that do not match an artifact value
  are reported as findings, never silently accepted.
* **Mechanism audit** — places every piece of evidence on the ladder
  `OBSERVATION → CORRELATION → CONTROLLED_COMPARISON → INTERVENTION → NECESSITY → SUFFICIENCY →
  RESCUE → CROSS_SETTING_REPLICATION`, then flags overreach: causal language without intervention,
  "establishes the mechanism" without necessity/sufficiency, single-setting generalisation,
  un-eliminated alternative explanations, dependency on one model/seed/scale.
* **AI style audit** — deterministic rules over: empty background, template phrases, repetition and
  n-gram self-similarity, buzzword density, unsupported causal/novelty language, generic statements,
  rhetorical adjective abuse, absence of researcher-specific reasoning, and mismatch with the real
  research history (e.g. "we hypothesised … and confirmed" when the timeline shows hypothesis *revision*).
* **Red team** — falsification questions, surviving alternative explanations, weakest experiment,
  single-model/single-seed dependencies, under-matched baselines, contaminable metrics, closest prior art,
  and the reviewer's counter-reading. Output: `RED_TEAM_REPORT`. It can only *report*.

### 4.12 Paper Compiler

Input (nothing else): verified evidence, approved claims, approved interpretations, literature evidence,
author notes, research decisions. Output sections: title, abstract, introduction, related work, method,
experiments, results, analysis, limitations, discussion, conclusion, references, appendix.

Every compiled sentence is a `GroundedSentence` carrying `claim_ids[]`, `evidence_ids[]`,
`numbers[] = NumberRef(value, unit, analysis_artifact, locator)`, `citations[] = CitationRef(paper_id, page, section)`.

Three post-compilation gates (the compiler refuses to emit a paper that fails them):

1. **Number grounding** — every numeral in the rendered text must match a `NumberRef` taken from an
   analysis artifact, within a declared tolerance. A numeral with no backing artifact is
   `UNGROUNDED_NUMBER` → compile blocked. *The writer cannot invent a number.*
2. **Claim grounding** — every load-bearing sentence resolves `sentence → claim → evidence → experiment → artifact`.
3. **Citation grounding** — every prior-work statement resolves `sentence → literature_claim → paper_id → page/section`,
   and a paper whose `fulltext_status != FULLTEXT_VERIFIED` may not be used for mechanism-level statements.

`PaperReadiness` reports seven *separate* dimensions (no single score):
evidence completeness, statistical completeness, mechanism evidence, literature coverage,
claim grounding, reproducibility, writing readiness.

### 4.13 Skill Intelligence Layer

`SkillCard`: `skill_id, name, description, source, repository, commit, license, dependencies[],
capabilities[], limitations[], tests[], examples[], benchmark, compatibility, quality_metrics,
known_failures[], status`.

Lifecycle: `DISCOVERED → CANDIDATE → SANDBOX → EVALUATED → VERIFIED → ACTIVE → DEGRADED → DEPRECATED`.
Hard gates: **no `ACTIVE` without a benchmark run**; **no `ACTIVE` if regression against the incumbent
fails**; **no generated skill is trusted** (synthesis output enters at `CANDIDATE`).

`SkillScore` weights: accuracy, coverage, citation correctness, claim calibration, reproducibility,
research relevance, failure handling, regression safety. There is deliberately **no "code runs" metric**.
Capability benchmark suites: literature (early/recent work, citation support, prior-art matrix,
missed terminology), experiment (confound, baseline matching, metric definition, paired/unpaired,
reproducibility), mechanism (correlation vs causation, missing intervention, necessity/sufficiency,
alternative explanation), writing (AI-tone removal, language-strength control, no new numbers,
evidence grounding, researcher voice).

### 4.14 Experiment OS

`Experiment` supports the full requested schema (question, hypothesis, IVs/DVs, treatment, control,
matched conditions, model, dataset, scale, optimizer, LR, seeds, n, budget, metrics + definitions,
code commit, config, raw artifacts, analysis artifacts, result, statistics, limitations) with
`status ∈ PLANNED | RUNNING | COMPLETED | FAILED | ABORTED`. Failed and aborted experiments are
**retained**: their status changes, their record never disappears.

---

## 5. Deterministic vs. LLM boundary

| Deterministic (never the LLM) | LLM (proposes, never decides) |
|---|---|
| hashing, artifact lineage, file existence | hypothesis generation |
| statistics: n/mean/std/SE/CI/effect size/p/tests/corrections | semantic synthesis across literature |
| seed counting, run dedup, config matching | interpretation of results |
| schema validation, revisions, state transitions | planning and task decomposition |
| citation metadata and DOI/arXiv normalisation | drafting prose |
| claim/evidence level derivation | suggesting claim wording |
| permission checks, conflict trust ranking | alternative explanations |
| style rules, buzzword density, grounding verification | literature query *expansion* (the query set is recorded and re-checked) |

The LLM layer is a thin adapter: `LLMProvider.complete(messages, **kwargs) -> LLMResponse` with
implementations for OpenAI-compatible endpoints (incl. local vLLM/Ollama), Anthropic, and a
deterministic `EchoProvider` for tests. No truth-path module imports `researchos.llm`
(enforced by a test that greps the import graph).

---

## 6. The ten invariants → enforcement → tests

| # | Invariant | Enforced in | Test |
|---|---|---|---|
| 1 | Agents cannot modify raw evidence | `kernel.guards.RESOURCE_GUARDS`, `evidence.registry` | `test_invariants.py::test_agent_cannot_modify_raw_evidence` |
| 2 | A local task cannot silently change the core question | `kernel.transitions`, `Task.priority` rules | `test_invariants.py::test_debug_task_cannot_promote_to_core` |
| 3 | The writer cannot create new numbers | `paper.grounding.verify_numbers` | `test_invariants.py::test_writer_cannot_invent_numbers` |
| 4 | Every claim has evidence | `claims.lifecycle` gates | `test_invariants.py::test_claim_requires_evidence` |
| 5 | Every literature claim has a source | `models.literature` validator | `test_invariants.py::test_literature_claim_requires_source` |
| 6 | `UNKNOWN` never becomes `FALSE` | `literature.prior_art` three-valued logic | `test_invariants.py::test_unknown_is_not_false` |
| 7 | External skills cannot mutate research state | `external_skill:*` principal has no write caps | `test_invariants.py::test_external_skill_cannot_mutate_state` |
| 8 | A skill upgrade cannot break existing benchmarks | `skills.lifecycle` regression gate | `test_invariants.py::test_skill_upgrade_requires_regression_pass` |
| 9 | Every experiment artifact traces to a source | `Provenance` + `ArtifactRef` re-hash | `test_invariants.py::test_artifact_traces_to_source` |
| 10 | Rejected claims never vanish | tombstone semantics + guards | `test_invariants.py::test_rejected_claim_is_retained` |

Additional enforced invariants:
`HYPOTHESIS → SUPPORTED` is rejected · event log tampering is detected ·
`CANDIDATE_GAP ≠ VERIFIED_NOVELTY` · `UNKNOWN` cells keep prior-art coverage honest ·
concurrent writes cannot silently drop a revision.

---

## 7. Delivery phases

**P0 (first, complete):** Research State · Task isolation · Research Import · Experiment object ·
Evidence object · Claim registry · basic provenance · literature search abstraction ·
citation/source tracking · paper grounding.
**P1:** literature graph · prior-art matrix · novelty audit · statistical audit · mechanism audit ·
skill registry · skill discovery · skill benchmark.
**P2:** skill auto-evolution · skill synthesis · dashboard · paper compiler · researcher voice ·
red-team loop · advanced reproducibility.

The ordering rule for the whole project: *never* build the pretty UI first, *never* build an autonomous
scientist demo first. First make **research state correct, evidence traceable, and local tasks unable
to drift**. When the UI does arrive (section 9), it is a *derivation* of that state, and the audit
console is not allowed to be the home screen.

---

## 9. Two views, two questions (the dashboard is a derivation, not a dump)

The first dashboard we built failed the way most research tooling fails: it rendered the *audit console*
— revision, event head, integrity counters, evidence maturity, skill health, timeline — as the home
screen. Everything was true and nothing was readable, because those fields answer a question the
researcher did not ask.

The fix is a split, and the split is an architectural rule rather than a styling choice:

| | **Research Cockpit** (`/cockpit`, `#/`) | **Audit Console** (`/dashboard`, `#/audit`) |
|---|---|---|
| Question | *Where is my research, and what do I do next?* | *Is this system trustworthy?* |
| Contains | phase, core question, current finding, what needs attention, the evidence→claim chain, paper readiness, the research map, where the project sits in its own process | revision, integrity, event head, artifact hashes, provenance completeness, conflicts, evidence maturity, skill health, timeline, import diagnostics, audit history, review queue |
| Forbidden | revision, hashes, event head, integrity counters, skill counters | — |
| Derived in | `researchos/cockpit.py` (deterministic, testable) | serialised models |
| Test | `test_cockpit.py`, and `test_dashboard.py::test_cockpit_payload_has_no_system_bookkeeping` asserts the forbidden fields are absent | `test_dashboard.py` |

The cockpit answers five questions and nothing else: *what am I researching · what do I know · what can I
not trust yet · why can I not continue · what do I do next.* Three consequences worth stating:

* **The phase is the earliest unmet gate**, not a mood: `QUESTION_UNCONFIRMED → IMPORT_REVIEW →
  CONFLICT_RESOLUTION → EVIDENCE_VERIFICATION → LITERATURE_AUDIT → EXPERIMENT_DESIGN → ANALYSIS_PENDING →
  CLAIM_VALIDATION → PAPER_COMPILATION → PAPER_READY`. Each phase carries the sentence that explains it,
  and the gate checklist shows done / current / pending so the state machine itself is visible.
* **Attention is derived, ordered and actionable.** One action per kind, severity first, each naming the
  exact command and *what it blocks*. "28 items in the review queue" becomes "decide 28 reconstructed
  items — nothing imported is a research fact until a human decides it".
* **The research map draws only links that exist** (claim→experiment declared links, structural
  belongs-to edges, and `SAME_SOURCE` for shared provenance — explicitly labelled as provenance, never as
  support). A claim no experiment points at is shown unlinked, and the cockpit raises that as a blocker in
  its own right, because a claim with no linked experiment can never leave `HYPOTHESIS`.

---

## 10. Reporting contract

Every module lands with: `WHAT WAS BUILT · WHY · DATA MODEL · INVARIANTS · TESTS · KNOWN LIMITATIONS ·
NEXT DEPENDENCY`. See `docs/architecture/` for the per-module notes.
