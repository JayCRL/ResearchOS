# Prior art: what ResearchOS adopts, modifies and rejects

*Scope: an engineering review of the open-source and published systems the spec named, judged only by
what ResearchOS needs: structured research state, traceable evidence, audited claims, measured
literature coverage, and skills that must earn their status. Facts about those projects come from their
own documentation and papers; the judgements are ours.*

This is a **living document**. Nothing here is a claim that these projects are deficient: most of them
solve a different problem well. The question each time is narrower — *does adopting this make research
state more correct, or does it put a language model on the truth path?*

---

## 1. Summary table

| System / project | What it is great at | Adopt | Modify | Reject |
|---|---|---|---|---|
| **Sakana "The AI Scientist"** ([paper](https://arxiv.org/abs/2408.06292), [overview](https://vectorinstitute.ai/the-ai-scientist-automated-research-life-cycle/)) | End-to-end idea→experiment→paper loop; automated review | Stage separation (idea / experiment / write / review) as an *audit* vocabulary; per-stage artifacts | Reviewer must become an auditor with **reference-free checks** (grounding, statistics, mechanism rung), not a taste model; every stage writes to state, not to a chat log | Autonomous end-to-end publication, LLM-as-reviewer as the final authority, "novelty" asserted by the writer |
| **STORM / Co-STORM** ([paper](https://ar5iv.labs.arxiv.org/html/2402.14207)) | Perspective-guided question asking; multi-agent knowledge curation; outline-driven writing | Perspective diversity as a *query-planning* input; outline-first structure | The outline must be generated from approved claims only, and each perspective must become a recorded query family; citations must resolve to page-level literature claims | Free-form article generation from retrieval; treating retrieved prose as knowledge without a locator |
| **PaperQA / PaperQA2** ([docs](https://docs.edisonscientific.com/paperqa), [pypi](https://pypi.org/project/paper-qa/)) | Evidence-grounded QA over full texts; citation-anchored answers; contradiction surfacing | Evidence objects with page/section provenance; contradiction-first behaviour; full-text-first stance | Contradictions must become **first-class Conflict records** with a deterministic trust order, not transient answer annotations | Treating an answer as the artifact: ResearchOS keeps evidence and claims, and compiles prose last |
| **Scientific Agent Skills** ([paper](https://arxiv.org/html/2609.00065v1)) | 163 authored procedures; `SKILL.md` with a constrained YAML header; three-tier *progressive disclosure*; explicit statement of what it does **not** measure | Skill-as-directory, versioned instruction file, progressive disclosure (small resident tier, large deferred tier), declared capability + provenance + compatibility, "reach is not use" honesty | Skills enter an at-most-**advisory** lifecycle: sandbox → benchmark → regression → ACTIVE, and external skills can only *propose* | Installing skills as trusted context; treating a passing structure as a scientific guarantee (their own limitations section says a selected skill "can still fail scientifically") |
| **scicomp-research-skills / AGENTS.md-style repos** ([example](https://raw.githubusercontent.com/a-attia/scicomp-research-skills/main/AGENTS.md)) | Concrete, practical research-engineering instructions; agent-readable conventions | Instruction files as reviewable text artifacts; convention documentation as a first-class deliverable | Pin to a commit + content hash, and require a declared capability ceiling before use | Vague trust: an unpinned instruction file is unreviewable, so ResearchOS records `commit`/`content_hash` or refuses to activate |
| **DVC / Sacred / MLflow** | Experiment tracking, config capture, artifact versioning, metric logging | Provenance fields (commit, config hash, seeds, environment, artifacts); run/param/metric triple; artifact lineage | Provenance must be **content-addressed and re-verified** at audit time, and attached to *claims*, not only to runs | A tracking server as the source of truth: ResearchOS keeps plain YAML/JSONL that git can diff and a human can read without our software |
| **Code Ocean** | Container-pinned reproducibility capsules | The idea that a run is only reproducible with its environment | Environment capture is recorded as provenance and *reported as incomplete when missing* | Hosted-only capsules as the unit of truth |
| **Semantic Scholar / arXiv / OpenReview / Crossref tooling** | Metadata, citation graphs, semantic neighbours | Provider-adapter interface; citation/reference expansion as a query family | Providers must be swappable and every result recorded with its raw payload; coverage measured per family | Any single provider hard-wired into the architecture; "we searched arXiv once" as coverage |
| **Zotero integrations** | Human curation of a personal library | Human-in-the-loop inclusion decisions (`include` flag, review queue) | The library is not the literature map: the map needs relations, clusters and coverage | Treating a bibliography as prior-art coverage (BibTeX metadata is not full-text knowledge) |
| **Causal / mechanistic interpretability tooling** | Interventions, ablations, necessity/sufficiency designs | The mechanism ladder (`observation → correlation → controlled → intervention → necessity/sufficiency → rescue → cross-setting`) as an explicit audit object | The ladder is *derived from declared design flags*, never from prose, and claim language is capped by it | Reporting a controlled comparison as "the mechanism" |
| **Agent-evaluation benchmarks (SWE-bench-style, ScienceAgentBench, SkillsBench)** | Fixed task suites, pass/fail scoring, regression discipline | Fixed benchmark tasks with required/forbidden findings, activation thresholds, task-level regression gates | Benchmarks are scored on **capability-specific findings**, not on whether code ran | "It executed" as a quality metric |

---

## 2. The four extractions that changed our design

1. **From PaperQA2: contradiction surfacing ⇒ the Conflict ledger.**
   Papers disagree, and a system that silently picks a winner is worse than one that says so. ResearchOS
   made this structural: `Conflict` records both sources with digests, applies the provenance trust order
   only as a *default*, and **an unresolved conflict blocks claim promotion**.

2. **From Scientific Agent Skills: progressive disclosure ⇒ bounded skill content, measured cost.**
   Their measurement — a small always-resident tier with the bulk deferred — is the right model for a
   skill *library*. ResearchOS adopts it and adds what the paper explicitly leaves open: task-level
   evaluation, a regression gate against the incumbent, and a hard rule that a generated skill is not a
   trusted skill.

3. **From STORM: perspective diversity ⇒ query families.**
   Multi-perspective question asking is a strong idea, but for *coverage* it has to be recorded. We
   turned perspectives into the `QueryFamily` enum and made a novelty verdict illegal below
   `COVERAGE_THRESHOLDS`. A search that did not run a family cannot license "no match found".

4. **From the tracking tools: provenance ⇒ verification, not decoration.**
   Logging a commit is not provenance unless something re-checks it. `ArtifactRef` re-hashes the file at
   audit time, and a changed byte becomes a **blocking conflict**, not a stale green check.

---

## 3. What we reject, and why (stated plainly)

* **LLM-as-final-authority on facts.** No model output may enter `research_state`, a claim status, a
  number, or a citation without a deterministic check. Enforced by `tests/test_layering_and_cli.py`.
* **"Autonomous scientist" as the product.** The product is correct, auditable research state. An agent
  loop is a consumer of that state, not the system of record.
* **Novelty claims from a keyword search.** Not expressible below the coverage thresholds, by model
  validator.
* **Free-form paper generation.** The compiler reads approved material and refuses ungrounded numerals.
* **Skill installation as trust.** External skills hold `propose.*` capabilities only.
* **A pretty UI first.** CLI and API first; the dashboard is a view over state that must already be right.

---

## 4. Two things we deliberately did *not* copy, and the cost

* **Container-per-run reproducibility (Code Ocean style).** We record environment and hashes but do not
  manage containers. Cost: `REPRODUCIBILITY` readiness can only report what provenance exists, and says
  so in its `unknowns`.
* **A tracking backend (MLflow/DVC).** We keep files. Cost: no query engine over runs; a SQLite index is
  planned as a *derived* cache, never a source of truth.

Both are recorded as readiness dimensions rather than hidden, which is the whole point: the system
reports the gap instead of assuming it away.

---

## 5. References

* Sakana AI, *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery* —
  [arXiv:2408.06292](https://arxiv.org/abs/2408.06292) ·
  [Vector Institute overview](https://vectorinstitute.ai/the-ai-scientist-automated-research-life-cycle/)
* Stanford, *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models* (STORM)
  — [arXiv:2402.14207](https://ar5iv.labs.arxiv.org/html/2402.14207)
* *PaperQA2* — [documentation](https://docs.edisonscientific.com/paperqa) ·
  [PyPI](https://pypi.org/project/paper-qa/)
* Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents* —
  [arXiv:2609.00065](https://arxiv.org/html/2609.00065v1)
* *scicomp-research-skills* `AGENTS.md` convention —
  [example repository file](https://raw.githubusercontent.com/a-attia/scicomp-research-skills/main/AGENTS.md)
* *Beyond static responses: multi-agent LLM systems for social science research* —
  [Nature HSSC](https://www.nature.com/articles/s41599-026-08832-2)
