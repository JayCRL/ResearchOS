# README

**English** | [简体中文](README.zh-CN.md)

**ResearchOS is a Research Operating System for AI/ML science.** Not a chat assistant, not an AI paper
writer. It makes **research direction, evidence, claims, literature context and skills** into structured
state that both humans and machines can audit.

> **Chat history is not research state.**
> **The LLM is not the source of scientific truth.**
> **A paper is a compilation artifact, not a source of truth.**

```
GLOBAL RESEARCH STATE → CURRENT TASK → AGENT EXECUTION → EXPERIMENT / ANALYSIS / LITERATURE
        → EVIDENCE → AUDIT → CLAIM UPDATE → RESEARCH STATE UPDATE → PAPER COMPILER
```

---

## Quick start

```bash
pip install -e ".[dev]"

# take over an existing project (the first experience that matters)
researchos import ./my-existing-research --into ./my-project

# see the research state, not a summary
researchos status
researchos dashboard

# work a bounded task, so a local problem stays local
researchos task create "explain the seed-3 metric drop" --purpose DEBUG --priority EXPLORATORY \
  --stop "metric explained or escalated to a human"

# claims, evidence and audits
researchos claim list
researchos claim audit          # shows which wording exceeds which evidence, with a calibrated rewrite
researchos evidence verify <evidence-id>
researchos audit stats
researchos mechanism --claim <claim-id>

# literature and novelty
researchos literature search "write placement in fast weights"
researchos novelty audit "nobody has compared placement under a matched-energy control"

# the project explains its own history, and attacks its own claims
researchos timeline                       # narrative: observation → hypothesis → experiment → revision → claim
researchos timeline ask --question "why was this control added?"
researchos redteam                        # RED TEAM REPORT: what would falsify this, and what is weakest

# the paper is compiled, gated, and refuses ungrounded numbers
researchos paper compile --out paper.md
researchos paper readiness

# skills earn their status, and upgrades are regression-gated
researchos skill benchmark --suite LITERATURE --seed-tasks
researchos skill evolve --from <skill-a>,<skill-b> --rule "every claim needs a locator" --findings findings.json
```

`researchos --help` lists every command. A step-by-step usage guide (with real output, what each refusal
means, a command cheat sheet and the Python API) is in [`docs/usage.zh-CN.md`](docs/usage.zh-CN.md) (Chinese).

### Dashboard: a cockpit, not a status dump

```bash
pip install -e ".[api]"
researchos api                 # → http://127.0.0.1:8765/
researchos api --enable-actions   # + human-in-the-loop buttons (local clients only)
```

One HTML file, **no build step, no npm, no CDN** — it must still open in five years, offline. It has
**two views**, and keeping them apart is a design rule rather than a styling choice:

| | **Research Cockpit** (`#/`, default) | **Audit Console** (`#/audit`) |
|---|---|---|
| Question | *Where is my research, and what do I do next?* | *Is this system trustworthy?* |
| Shows | research phase, core question, what is currently known, **what needs your attention** (severity-ordered, each with the exact command and what it blocks), the evidence → claim chain (why a claim cannot move up), paper readiness (*Not ready* + reasons, seven dimensions behind a disclosure), the **research map**, and where the project sits in its own process | revision, integrity, event head, hashes, provenance, conflicts, evidence maturity, skill health, timeline, import diagnostics, audit history, review queue |
| Never shows | revision, hashes, event head, integrity counters, skill counters | — |

The home screen answers five questions and nothing else: *what am I researching · what do I know · what
can I not trust yet · why can I not continue · what do I do next.* The phase is the **earliest unmet
gate** rather than a mood, and the research map draws **only links that exist** — a claim no experiment
points at is shown unlinked *and* raised as a blocker, because such a claim can never leave `HYPOTHESIS`.

The dashboard is a **view**: `.researchos/` stays the source of truth. Writes are off unless you pass
`--enable-actions`, and even then they run as the **human principal through the same kernel gate as the
CLI** — a claim with missing evidence is still refused, a stale transition is still refused, and every
click is recorded as a `ui.action` event so it is distinguishable from a shell command.

---

## What it does about the six problems it exists for

| Problem | Mechanism in the codebase |
|---|---|
| **Context pollution / research-line drift** | `research_state.yaml` is the source of truth; `Task` carries a priority, a stop condition and an allowed-action set; guarded state (core question, core claims, priorities, non-goals, scope) changes **only** through an approved `StateTransitionRequest`, and a `DEBUG`/`EXPLORATORY` task cannot even file one (`kernel/state.py`, `kernel/transitions.py`, `kernel/tasks.py`) |
| **Invented facts, numbers and results** | Numbers live in `Analysis` artifacts; the paper compiler builds a `NumberRef` pool from them and re-extracts every numeral from the rendered text; an unmatched numeral is a **blocking** `UNGROUNDED_NUMBER` (`paper/grounding.py`) |
| **Insufficient understanding of existing research** | `researchos import` runs Research Archaeology: scan → classify → extract → conflict detection → research-state reconstruction → **human review queue**. It never imports a claim above `HYPOTHESIS` (`importer/`) |
| **Narrow literature search** | `QueryFamily` forces exact, synonym, historical, mechanistic, functional, neighbouring-community and recent terminology families; coverage is measured per family and provider, and a novelty verdict is illegal below the thresholds (`literature/`) |
| **"Written by AI" papers** | Claims are compiled, not generated: approved claims + verified evidence + analysis numbers + literature claims with locators. A deterministic style auditor flags empty background, template language, repeated n-grams, buzzword density, unsupported causal/novelty language and **history mismatch** (`paper/style_audit.py`, `claims/language.py`) |
| **Skills that never improve** | A Skill Meta-System: discovery → registry → sandbox → benchmark → regression → ACTIVE, with the rule that a generated skill is **not** a trusted skill and an upgrade that breaks a previously passing benchmark is refused. `researchos skill evolve` runs the whole loop (`skills/evolution.py`) |

**What ResearchOS does not claim:** it does not guarantee that a paper is correct, that a claim is novel,
that a mechanism is proven, or that an experiment is reproducible. It makes those questions *answerable
and auditable*, and it puts the uncertainty on the screen instead of in the prose.

---

## Architecture in one screen

```
┌─────────────────────── RESEARCHOS KERNEL ───────────────────────┐
│ Global Research State · Permission Gate · State Transitions ·    │
│ Provenance · Task Boundaries · Claim/Evidence/Literature/Skill   │
│ State · append-only hash-chained Event Log · Conflict Ledger     │
└───────┬───────────────┬───────────────┬──────────────┬───────────┘
        ▼               ▼               ▼              ▼
   Literature OS    Evidence OS    Experiment OS    Skill OS
        └───────────────┴───────────────┴──────────────┘
                        ▼
              AGENT OS (15 bounded agents)
                        ▼
   AUDITORS (statistical · mechanism · novelty · style · red team)
                        ▼
                 CLAIM REGISTRY
                        ▼
                 PAPER COMPILER
```

Full design and rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md) ·
invariants map: [`docs/invariants.md`](docs/invariants.md) ·
prior-art review (adopt / modify / reject): [`docs/prior_art_review.md`](docs/prior_art_review.md)

---

## On-disk layout: everything is reviewable text

```
.researchos/
├── project.yaml
├── state/
│   ├── research_state.yaml      ★ GLOBAL RESEARCH STATE (git-diffable source of truth)
│   ├── events.jsonl             ★ append-only, SHA-256 hash-chained history
│   ├── tasks/                   bounded tasks with priorities and stop conditions
│   ├── transitions/             every proposed and approved direction change
│   └── decisions/               why things changed (never rewritten)
├── claims/  experiments/  evidence/{,raw}  analysis/
├── literature/{papers,graph,claims,search,prior_art,novelty,gaps}
├── skills/{registry,versions,benchmarks,sandbox,gaps,deprecated}
├── audits/  conflicts/  review/  notes/  timeline/  paper/{artifacts,audits}
└── cache/                       disposable, rebuildable — never a source of truth
```

---

## The fifteen agents

Agents are **not** autonomous loops with a system prompt. Each is a named principal with a fixed
capability set; it can always *propose*, and the kernel decides. Two examples of what that means in
practice:

* `paper_writer` cannot propose a claim, and cannot produce a numeral that no analysis artifact
  produced.
* `researchos agent list` prints, for every agent, the capabilities it does **not** hold — including
  `claim.approve`, `state.transition.approve`, `state.core.write` and `skill.activate`, which no agent
  holds.

```
researchos agent list
```

---

## Testing

```bash
python -m pytest -q                 # 180 tests
python -m pytest -m invariant -q    # the ten invariants only
```

The read-only HTTP surface (optional extra) and the researcher-voice view:

```bash
pip install -e ".[api]"
researchos api --port 8765          # read-only: writes need an explicit principal, so they are not on HTTP
researchos paper voice              # the real research arc + the researcher's own voice fingerprint
```

The invariant suite is the product, not the proof of it. It asserts, among other things, that an agent
cannot modify raw evidence, that a debugging task cannot redefine the core question, that the writer
cannot invent a number, that `HYPOTHESIS → SUPPORTED` is refused, that `UNKNOWN` never becomes `FALSE`,
that an external skill cannot mutate research state, that a skill upgrade cannot break a benchmark, that
a changed artifact is detected by re-hashing, and that a rejected claim never disappears.

---

## Status and honest limitations

* **Implemented and tested:** kernel (state, permissions, transitions, provenance, event log, conflicts),
  data models, Research Import, Evidence OS, Claim OS (lifecycle + language calibration), Literature OS
  (providers, query planning, coverage, graph, prior art, novelty audit), analysis (statistics,
  statistical audit, mechanism audit), Paper Compiler with grounding gates + style audit + readiness,
  agents, the full CLI, optional LLM adapters.
* **Not implemented:** web UI, PDF-heavy import beyond text extraction, container-level reproducibility,
  a SQLite read index, skill *synthesis* quality evaluation (the scaffolding and gates exist; no
  published synthesis benchmark yet), automatic git-history mining beyond commit subjects.
* **Known limitations:** import heuristics are deterministic and therefore conservative (they miss
  things rather than invent them); the style auditor is rule-based and will miss novel AI phrasing;
  calibration can over-weaken a sentence rather than choose the exact permitted strength; the
  hash-chained log cannot detect *tail* truncation without an external anchor (commit the head digest).

*ResearchOS does not make research correct. It makes research direction, evidence and claims inspectable —
and refuses to let a language model decide what is true.*
