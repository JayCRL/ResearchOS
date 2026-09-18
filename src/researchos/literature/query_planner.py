"""Query planning: one question is never one search.

Why this module exists
----------------------
Keyword search over a research idea fails in a small number of predictable ways. The exact phrase
the author invented returns nothing, and the author concludes "nobody has done this" without ever
trying the synonym, the older name the field used before it renamed itself, the mechanism-level
description of the same idea, or the neighbouring community that solved the same problem under
different vocabulary. Every one of those failure modes is a query *family* here, and
:data:`~researchos.models.literature.REQUIRED_FAMILIES` lists the families a search must exercise
before a no-match verdict is even expressible.

Three rules make a plan trustworthy:

1. **Template expansion is deterministic.** Synonyms, historical terms, mechanism and functional
   equivalents come from frozen lexicons (:data:`SYNONYM_LEXICON`, :data:`HISTORICAL_TERMS`,
   :data:`MECHANISM_EQUIVALENTS`, :data:`FUNCTIONAL_EQUIVALENTS`, :data:`RECENT_TERMS`) and every
   generated query carries a ``rationale`` naming the family it serves. The same target always
   yields the same plan, so "our search was thin" is reproducible and reviewable by hand.
2. **The LLM may expand, never grade.** An injected ``expand_with_llm`` callable may add queries,
   they are labelled :attr:`QueryFamily.SEMANTIC_NEIGHBOR`, and its use is recorded on the plan as
   ``llm_used_for_expansion``. It cannot satisfy a required family and it never touches coverage,
   which is computed only from *executed* queries and papers actually ingested.
3. **Ingestion copies metadata, never interpretation.** :meth:`QueryPlanner.ingest` marks every new
   paper ``ABSTRACT_LEVEL_ONLY`` (or ``UNKNOWN``) and leaves the fulltext-only fields empty,
   because the literature model rejects a paper that pretends to know a mechanism from an abstract.
   A paper already known to be ``FULLTEXT_VERIFIED`` is never rewritten by a later provider hit.

Why :meth:`QueryPlanner.plan` does not persist: a plan is a *proposal* and holds no literature
record. It is written to ``kernel.query_plans`` the moment it is executed (or expanded from seeds),
which is the first point at which it has results to be accountable for. A plan that was never
executed leaves no trace in the event log, and a query plan can never be mistaken for a search that
happened.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from ..kernel.errors import LiteratureError
from ..kernel.permissions import Cap, Principal
from ..models.common import FulltextStatus, utcnow
from ..models.literature import (
    REQUIRED_FAMILIES,
    LiteratureCoverage,
    LiteraturePaper,
    PlannedQuery,
    ProviderKind,
    ProviderRecord,
    QueryFamily,
    QueryPlan,
)
from .coverage import compute_coverage
from .providers.base import LiteratureProvider, ProviderError, tokenise

# --------------------------------------------------------------------------------------
# Lexicons: what the field calls the same thing
# --------------------------------------------------------------------------------------

#: Canonical term -> the terms other papers use for the same thing.
#: Keys are matched against the target's tokens (all key tokens must appear), so a key such as
#: ``"fast weights"`` matches any target that mentions both *fast* and *weights*.
SYNONYM_LEXICON: dict[str, tuple[str, ...]] = {
    "fast weights": (
        "fast weight programmers",
        "rapidly adaptive weights",
        "dynamic weight plasticity",
        "associative memory weights",
    ),
    "writeback": (
        "write-back",
        "memory writing into parameters",
        "state consolidation into weights",
        "parameter consolidation",
    ),
    "retention": (
        "catastrophic forgetting",
        "knowledge retention",
        "memory persistence",
        "forgetting mitigation",
    ),
    "selection": (
        "routing",
        "gating",
        "content-based addressing",
        "retrieval keying",
    ),
    "memory": (
        "external memory",
        "memory-augmented network",
        "key-value memory store",
        "episodic store",
    ),
    "replay": (
        "experience replay",
        "rehearsal",
        "offline replay",
        "memory replay",
    ),
    "sleep": (
        "offline consolidation phase",
        "sleep-like consolidation",
        "rest-period replay",
        "downtime consolidation",
    ),
    "continual learning": (
        "lifelong learning",
        "sequential task learning",
        "incremental learning",
        "non-stationary learning",
    ),
    "sparsity": (
        "sparse activation",
        "top-k activation",
        "conditional computation",
        "mixture of experts routing",
    ),
}

#: Canonical term -> the *historical* names, i.e. what the idea was called before the current name.
#: This family exists because a 1992 result does not use 2024 vocabulary, and a search that only
#: speaks 2024 vocabulary will report it as absent.
HISTORICAL_TERMS: dict[str, tuple[str, ...]] = {
    "fast weights": (
        "dynamic link matching",
        "rapid weight plasticity",
        "transient weight storage",
    ),
    "memory": (
        "Hopfield network",
        "correlation matrix memory",
        "Willshaw model",
        "sparse distributed memory",
    ),
    "replay": (
        "hippocampal replay",
        "dual representation theory",
        "consolidation during rest",
    ),
    "selection": (
        "mixture of experts",
        "product of experts",
        "expert gating network",
    ),
    "retention": (
        "interference theory",
        "catastrophic interference",
        "stability-plasticity dilemma",
    ),
}

#: Communities that solve the same problem with a different vocabulary. A search that stays inside
#: one community cannot know what the neighbouring one already published.
NEIGHBOUR_COMMUNITIES: tuple[str, ...] = (
    "computational neuroscience",
    "cognitive science",
    "memory-augmented neural networks",
    "information retrieval",
    "databases and storage systems",
    "reinforcement learning",
    "systems for machine learning",
    "biologically plausible learning",
)

#: Canonical term -> the *mechanism-level* description of the same idea. Searching the mechanism,
#: not the label, is what finds work published under an unrelated name.
MECHANISM_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "writeback": (
        "local learning rule updating parameters from activations",
        "gradient-based consolidation of recurrent state",
        "parameter update rule with memory term",
    ),
    "selection": (
        "learned routing function over memory slots",
        "attention-based selection of stored items",
        "top-k gating over experts",
    ),
    "sleep": (
        "periodic offline optimisation phase",
        "interleaved consolidation pass over stored data",
    ),
    "memory": (
        "key-value store with learned addressing",
        "recurrent state carried across episodes",
        "external differentiable memory",
    ),
    "fast weights": (
        "second-order weight matrix updated at inference",
        "activations used as temporary weights",
    ),
}

#: Canonical term -> *functional* equivalents: a different mechanism that achieves the same effect.
FUNCTIONAL_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "writeback": (
        "knowledge distillation into parameters",
        "context window instead of parameter change",
        "memory compression into weights",
    ),
    "retention": (
        "regularisation against parameter drift",
        "rehearsal of earlier tasks",
        "task-specific parameter isolation",
    ),
    "memory": (
        "retrieval-augmented generation",
        "long context attention",
        "cache of past activations",
    ),
    "fast weights": (
        "test-time adaptation",
        "in-context learning without weight updates",
        "state space model recurrence",
    ),
}

#: Canonical term -> terminology that is *new* (and therefore under-cited). A recent paper about an
#: old idea cites recent work, so an old-vocabulary search misses it, and vice versa.
RECENT_TERMS: dict[str, tuple[str, ...]] = {
    "fast weights": (
        "test-time training",
        "state space models",
        "linear attention",
    ),
    "selection": (
        "sparse mixture of experts",
        "token routing",
        "conditional compute",
    ),
    "retention": (
        "long-context forgetting",
        "context rot",
        "knowledge editing",
    ),
    "sleep": (
        "offline reinforcement learning consolidation",
        "generative replay at scale",
    ),
}

#: How many queries one family may contribute. Keeps a plan readable and a search bounded; the
#: *number of families* is what protects against blind spots, not the number of synonyms.
MAX_QUERIES_PER_FAMILY: int = 3

#: LLM-proposed queries are capped and always labelled :attr:`QueryFamily.SEMANTIC_NEIGHBOR`.
MAX_LLM_QUERIES: int = 5

#: A callable that proposes extra query strings: ``(target, families) -> ["...", ...]``.
ExpandWithLLM = Callable[[str, tuple[QueryFamily, ...]], Sequence[str]]

#: Families whose execution is a citation-graph traversal, not a keyword search. ``execute`` records
#: them as planned but never sends them to a search endpoint; :meth:`QueryPlanner.expand_from_seeds`
#: is the only thing that can mark them executed.
GRAPH_FAMILIES: tuple[QueryFamily, ...] = (
    QueryFamily.CITATION_EXPANSION,
    QueryFamily.REFERENCE_EXPANSION,
)

_FAMILY_RATIONALE: Mapping[QueryFamily, str] = {
    QueryFamily.EXACT_TERMINOLOGY: (
        "the verbatim phrase, quoted so the engine treats it as a phrase rather than a bag of words"
    ),
    QueryFamily.SYNONYMS: (
        "the same idea under the words other papers use; a single keyword misses renamed work"
    ),
    QueryFamily.HISTORICAL_TERMINOLOGY: (
        "the idea's earlier names; pre-rename literature does not use current vocabulary"
    ),
    QueryFamily.MECHANISTIC_EQUIVALENTS: (
        "the mechanism rather than the label, which finds work published under an unrelated name"
    ),
    QueryFamily.FUNCTIONAL_EQUIVALENTS: (
        "a different mechanism with the same function — the strongest form of prior art"
    ),
    QueryFamily.NEIGHBORING_COMMUNITIES: (
        "the same problem solved in a neighbouring community with different vocabulary"
    ),
    QueryFamily.RECENT_TERMINOLOGY: (
        "current terminology and year-bounded probes, where under-cited new work lives"
    ),
    QueryFamily.SEMANTIC_NEIGHBOR: (
        "an LLM-proposed expansion: useful, but never a substitute for a required family"
    ),
    QueryFamily.CITATION_EXPANSION: (
        "papers that cite the seeds, reached by citation-graph expansion rather than keyword search"
    ),
    QueryFamily.REFERENCE_EXPANSION: (
        "papers the seeds cite, reached by reference-graph expansion rather than keyword search"
    ),
    QueryFamily.AUTHOR_VENUE: (
        "the authors and venues that publish on this problem, which keyword search under-ranks"
    ),
}


# --------------------------------------------------------------------------------------
# Lexicon matching
# --------------------------------------------------------------------------------------


def _match_variants(target_tokens: frozenset[str], table: Mapping[str, tuple[str, ...]]) -> list[str]:
    """Variants of every key whose tokens all appear in the target. Sorted, deduplicated.

    Sorted key iteration and dedup are what make a plan reproducible: two runs on the same target
    produce byte-identical query lists, so a reviewer can diff two searches.
    """
    out: list[str] = []
    for key in sorted(table):
        key_tokens = tokenise(key)
        if not key_tokens or not all(token in target_tokens for token in key_tokens):
            continue
        for variant in table[key]:
            if variant not in out:
                out.append(variant)
    return out


def _family_texts(
    target: str,
    target_tokens: frozenset[str],
    table: Mapping[str, tuple[str, ...]],
    *,
    template: str,
) -> list[str]:
    """Render one query per matched variant, capped and deduplicated in a deterministic order."""
    variants = _match_variants(target_tokens, table)
    texts: list[str] = []
    for variant in variants:
        text = template.format(target=target, term=variant)
        if text not in texts:
            texts.append(text)
        if len(texts) >= MAX_QUERIES_PER_FAMILY:
            break
    return texts


class QueryPlanner:
    """Builds query plans, executes them against providers, and ingests what comes back.

    The planner is the only component that decides *what was searched*; coverage accounting
    (:mod:`researchos.literature.coverage`) and the novelty verdict
    (:mod:`researchos.literature.novelty`) only read the result. Keeping those three apart is what
    makes "we searched broadly" a measurable claim instead of an assertion.
    """

    def __init__(self, kernel) -> None:
        self.kernel = kernel

    # ================================================================== planning

    def plan(
        self,
        principal: Principal,
        target: str,
        *,
        target_refs: Sequence[str] = (),
        providers: Sequence[LiteratureProvider] = (),
        seed_paper_ids: Sequence[str] = (),
        max_expansion_rounds: int = 2,
        expand_with_llm: ExpandWithLLM | None = None,
    ) -> QueryPlan:
        """Draft a search plan for ``target``. Read-only: nothing is persisted here.

        Every member of :data:`REQUIRED_FAMILIES` gets at least one query, plus a citation-expansion
        placeholder, and an unknown target (no lexicon key matches) still gets all of them — the
        fallback is a real probe, because a field the planners have never heard of is exactly the
        field whose vocabulary we do not know.
        """
        principal.require(Cap.LITERATURE_READ, "literature.query_plan")
        cleaned = (target or "").strip()
        if not cleaned:
            raise LiteratureError("a query plan needs a non-empty target")

        kinds = sorted({provider.kind for provider in providers}, key=lambda kind: kind.value)
        queries = self._templated_queries(cleaned, kinds)
        notes: list[str] = [
            "deterministic template expansion from the frozen lexicons; every query records the "
            "family it serves",
            "the citation-expansion query is a placeholder: execute() never sends it to a search "
            "endpoint, expand_from_seeds() is what runs it",
        ]
        llm_used = False
        if expand_with_llm is not None:
            extra, llm_used, llm_note = self._llm_queries(cleaned, kinds, expand_with_llm)
            queries.extend(extra)
            notes.append(llm_note)

        return QueryPlan(
            target=cleaned,
            target_refs=list(target_refs),
            created_by=principal.name,
            seed_paper_ids=list(seed_paper_ids),
            queries=queries,
            providers=kinds,
            max_expansion_rounds=max(0, int(max_expansion_rounds)),
            notes=notes,
            llm_used_for_expansion=llm_used,
        )

    def _templated_queries(self, target: str, kinds: Sequence[ProviderKind]) -> list[PlannedQuery]:
        """One query list, in family order, with a rationale on every single entry."""
        tokens = frozenset(tokenise(target))
        year = utcnow().year
        specs: list[tuple[QueryFamily, list[str], str]] = [
            (
                QueryFamily.EXACT_TERMINOLOGY,
                [f'"{target}"'],
                "verbatim phrase",
            ),
            (
                QueryFamily.SYNONYMS,
                _family_texts(target, tokens, SYNONYM_LEXICON, template="{target} {term}"),
                "synonym expansion",
            ),
            (
                QueryFamily.MECHANISTIC_EQUIVALENTS,
                _family_texts(target, tokens, MECHANISM_EQUIVALENTS, template="{target} {term}"),
                "mechanism-level expansion",
            ),
            (
                QueryFamily.FUNCTIONAL_EQUIVALENTS,
                _family_texts(target, tokens, FUNCTIONAL_EQUIVALENTS, template="{target} {term}"),
                "functional-equivalence expansion",
            ),
            (
                QueryFamily.HISTORICAL_TERMINOLOGY,
                _family_texts(target, tokens, HISTORICAL_TERMS, template="{target} {term}"),
                "historical-terminology expansion",
            ),
            (
                QueryFamily.RECENT_TERMINOLOGY,
                _family_texts(target, tokens, RECENT_TERMS, template="{target} {term}")
                + [f"{target} {year}", f"{target} {year - 1}"],
                "recent-terminology and year-bounded probes",
            ),
        ]
        queries: list[PlannedQuery] = []
        for family, texts, label in specs:
            fallback = f"{target} {_FALLBACK_SUFFIX[family]}"
            chosen = [text for text in texts if text][:MAX_QUERIES_PER_FAMILY] or [fallback]
            rationale = (
                f"{_FAMILY_RATIONALE[family]} ({label}; "
                + ("no lexicon key matched the target, so a generic probe keeps the family present"
                   if chosen == [fallback]
                   else f"{len(chosen)} template(s) matched the target")
                + ")"
            )
            for text in chosen:
                queries.append(
                    PlannedQuery(family=family, text=text, rationale=rationale, providers=list(kinds))
                )
        queries.append(
            PlannedQuery(
                family=QueryFamily.CITATION_EXPANSION,
                text=f"citation-graph expansion from seeds of: {target}",
                rationale=_FAMILY_RATIONALE[QueryFamily.CITATION_EXPANSION],
                providers=list(kinds),
            )
        )
        return queries

    def _llm_queries(
        self,
        target: str,
        kinds: Sequence[ProviderKind],
        expand_with_llm: ExpandWithLLM,
    ) -> tuple[list[PlannedQuery], bool, str]:
        """Ask the injected callable for extra queries, and record honestly what happened.

        A failing or empty expansion is a *note*, never an error: the deterministic plan is the
        auditable artefact, and the optional expansion must not be able to break a search run.
        """
        try:
            proposed = [str(item).strip() for item in expand_with_llm(target, REQUIRED_FAMILIES)]
        except Exception as exc:  # noqa: BLE001 - an optional expansion must never abort a search
            return [], False, f"llm expansion failed and was ignored: {type(exc).__name__}: {exc}"
        queries: list[PlannedQuery] = []
        seen: set[str] = set()
        for text in proposed:
            if not text or text in seen:
                continue
            seen.add(text)
            queries.append(
                PlannedQuery(
                    family=QueryFamily.SEMANTIC_NEIGHBOR,
                    text=text,
                    rationale=_FAMILY_RATIONALE[QueryFamily.SEMANTIC_NEIGHBOR],
                    providers=list(kinds),
                )
            )
            if len(queries) >= MAX_LLM_QUERIES:
                break
        note = (
            f"llm expansion contributed {len(queries)} SEMANTIC_NEIGHBOR quer"
            f"{'y' if len(queries) == 1 else 'ies'}; it cannot satisfy a required family and is "
            "excluded from coverage accounting"
        )
        return queries, True, note

    # ================================================================== execution

    def execute(
        self,
        principal: Principal,
        plan: QueryPlan,
        providers: Sequence[LiteratureProvider],
        *,
        limit: int = 20,
        task_id: str | None = None,
    ) -> QueryPlan:
        """Run every keyword query in ``plan`` and ingest what the providers return.

        A provider that is unavailable or that raises is *recorded on the query* (``error``) rather
        than aborting the run: a search that lost one source is worth finishing and reporting, and
        the missing source then shows up as thin coverage. Graph families are left unexecuted on
        purpose — keyword-searching a placeholder would fabricate coverage.
        """
        principal.require(Cap.LITERATURE_WRITE, "literature.query_plan.execute")
        principal.require(Cap.LITERATURE_PROVIDER_USE, "literature.provider")
        provider_list = list(providers)
        kinds = sorted({provider.kind for provider in provider_list}, key=lambda kind: kind.value)

        before = set(self.kernel.papers.ids())
        for query in plan.queries:
            if query.executed or query.family in GRAPH_FAMILIES:
                continue
            records: list[ProviderRecord] = []
            errors: list[str] = []
            for provider in provider_list:
                if not provider.available():
                    errors.append(f"{provider.name}: unavailable")
                    continue
                try:
                    records.extend(provider.search(query.text, limit=limit))
                except ProviderError as exc:
                    errors.append(f"{provider.name}: {exc}")
            query.executed = True
            query.executed_at = utcnow()
            query.result_count = len(records)
            query.providers = list(kinds)
            query.error = "; ".join(errors) if errors else None
            if not provider_list:
                query.error = "no provider configured: query planned but never sent"
            ingested = self.ingest(principal, records) if records else []
            query.new_paper_ids = sorted(
                {paper.paper_id for paper in ingested if paper.paper_id not in before}
            )
            before.update(query.new_paper_ids)

        plan.providers = kinds
        self.kernel.query_plans.save(plan)
        self.kernel.events.append(
            "literature.queries_executed",
            actor=principal.name,
            task_id=task_id,
            payload={
                "query_plan_id": plan.query_plan_id,
                "target": plan.target[:200],
                "queries_executed": len(plan.executed_queries()),
                "providers": [kind.value for kind in kinds],
                "papers_after": self.kernel.papers.count(),
            },
        )
        return plan

    def ingest(
        self, principal: Principal, records: Sequence[ProviderRecord]
    ) -> list[LiteraturePaper]:
        """Turn provider records into papers, deduplicated and stripped of interpretation.

        Deduplication is by :meth:`LiteraturePaper.identity_key` (DOI, then arXiv id, then title), so
        the same paper found through two providers becomes one row with two attestations — counting
        it twice would inflate ``papers_screened``, the number a novelty verdict leans on.

        The first record wins for anything already known; a later provider hit only fills gaps and
        adds its own attribution. ``fulltext_status`` is *never* touched after the record is created:
        a hit without an abstract stays ``UNKNOWN``, a verified full text stays verified, and no
        merge can promote a paper to ``FULLTEXT_VERIFIED``. Mechanism-level fields are left empty
        because the model forbids them below full text.
        """
        principal.require(Cap.LITERATURE_WRITE, "literature.paper")
        store = self.kernel.papers
        by_key: dict[str, LiteraturePaper] = {}
        for existing in store.all():
            by_key.setdefault(existing.identity_key(), existing)

        out: list[LiteraturePaper] = []
        seen: set[str] = set()
        created = 0
        updated = 0
        for record in records:
            key = record.identity_key()
            if key in seen:
                continue
            seen.add(key)
            existing = by_key.get(key)
            if existing is None:
                paper = _paper_from_record(record)
                store.save(paper)
                by_key[key] = paper
                out.append(paper)
                created += 1
                continue
            merged = _merge_record(existing, record)
            if merged is not existing:
                store.save(merged)
                by_key[key] = merged
                updated += 1
            out.append(merged)

        if records:
            self.kernel.events.append(
                "literature.papers_ingested",
                actor=principal.name,
                payload={
                    "records": len(records),
                    "created": created,
                    "updated": updated,
                    "providers": sorted({record.provider.value for record in records}),
                },
            )
        return out

    # ================================================================== expansion

    def expand_from_seeds(
        self,
        principal: Principal,
        plan: QueryPlan,
        providers: Sequence[LiteratureProvider],
        *,
        rounds: int = 1,
        limit: int = 20,
    ) -> QueryPlan:
        """Walk the citation graph outward from the plan's seeds, ``rounds`` hops at a time.

        This is the only path that marks a graph family executed, and it is bounded by
        ``plan.max_expansion_rounds`` because citation expansion grows very fast: two hops from ten
        seeds can be thousands of papers, and a search that silently becomes unbounded is a search
        nobody can describe in a paper. When no seeds were given, the papers the plan already found
        become the seeds, so execute-then-expand is a valid two-step workflow.
        """
        principal.require(Cap.LITERATURE_WRITE, "literature.query_plan.expand")
        principal.require(Cap.LITERATURE_PROVIDER_USE, "literature.provider")
        provider_list = list(providers)

        allowed = max(0, plan.max_expansion_rounds - plan.expansion_rounds)
        planned_rounds = max(0, int(rounds))
        if planned_rounds > allowed:
            plan.notes.append(
                f"expansion capped at {allowed} round(s) by max_expansion_rounds="
                f"{plan.max_expansion_rounds} (requested {planned_rounds})"
            )
        plan_rounds = min(planned_rounds, allowed)
        if plan_rounds == 0:
            plan.notes.append("citation expansion skipped: no expansion rounds left in this plan")
            return plan

        frontier = self._seed_frontier(plan)
        if not frontier:
            plan.notes.append("citation expansion skipped: the plan has no resolvable seed papers")
            return plan

        citation_query = self._graph_query(plan, QueryFamily.CITATION_EXPANSION)
        reference_query = self._graph_query(plan, QueryFamily.REFERENCE_EXPANSION)
        new_ids: set[str] = set()
        for round_index in range(plan_rounds):
            round_papers: list[LiteraturePaper] = []
            for paper in frontier:
                for provider in provider_list:
                    if not provider.available():
                        continue
                    identifier = _seed_identifier(paper)
                    if not identifier:
                        continue
                    try:
                        seed_record = provider.fetch(identifier)
                        if seed_record is None:
                            continue
                        references, citations = provider.expand(seed_record)
                    except ProviderError as exc:  # a provider failure is recorded, never fatal
                        plan.notes.append(f"expansion via {provider.name} failed: {exc}")
                        continue
                    round_papers.extend(self.ingest(principal, (citations, references)[0]))
                    round_papers.extend(self.ingest(principal, references))
            fresh = self._unseen(plan, round_papers, new_ids)
            plan.expansion_rounds += 1
            new_ids.update(paper.paper_id for paper in fresh)
            if citation_query is not None:
                citation_query.executed = True
                citation_query.executed_at = utcnow()
                citation_query.result_count += len(fresh)
                citation_query.new_paper_ids = sorted(
                    set(citation_query.new_paper_ids) | {p.paper_id for p in fresh}
                )
            if reference_query is not None:
                reference_query.executed = True
                reference_query.executed_at = utcnow()
                reference_query.result_count += len(fresh)
                reference_query.new_paper_ids = sorted(
                    set(reference_query.new_paper_ids) | {p.paper_id for p in fresh}
                )
            plan.notes.append(
                f"expansion round {round_index + 1}: {len(frontier)} seed(s) -> "
                f"{len(fresh)} new paper(s)"
            )
            frontier = sorted(fresh, key=lambda item: item.paper_id)
            if not frontier:
                break

        self.kernel.query_plans.save(plan)
        self.kernel.events.append(
            "literature.queries_expanded",
            actor=principal.name,
            payload={
                "query_plan_id": plan.query_plan_id,
                "rounds": plan.expansion_rounds,
                "new_papers": len(new_ids),
                "papers_after": self.kernel.papers.count(),
            },
        )
        return plan

    def _seed_frontier(self, plan: QueryPlan) -> list[LiteraturePaper]:
        ids: list[str] = list(plan.seed_paper_ids)
        if not ids:
            for query in plan.executed_queries():
                ids.extend(query.new_paper_ids)
        papers: list[LiteraturePaper] = []
        for paper_id in sorted(set(ids)):
            paper = self.kernel.papers.get(paper_id)
            if paper is not None:
                papers.append(paper)
        return papers

    def _graph_query(self, plan: QueryPlan, family: QueryFamily) -> PlannedQuery | None:
        for query in plan.queries:
            if query.family is family:
                return query
        return None

    @staticmethod
    def _unseen(
        plan: QueryPlan, papers: Sequence[LiteraturePaper], already: set[str]
    ) -> list[LiteraturePaper]:
        known = {paper_id for paper_id in already}
        for query in plan.queries:
            known.update(query.new_paper_ids)
        out: list[LiteraturePaper] = []
        seen: set[str] = set()
        for paper in papers:
            if paper.paper_id in known or paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            out.append(paper)
        return sorted(out, key=lambda item: item.paper_id)

    # ================================================================== coverage

    def compute_coverage(self, plan: QueryPlan) -> LiteratureCoverage:
        """Coverage of *this* plan, measured over the papers this plan actually touched.

        Read-only and principal-free: it recomputes what the plan's own record already contains
        (executed queries, families, providers, ingested papers) and never consults the LLM or the
        network. Anything else would make the number a novelty verdict rests on unverifiable.
        """
        papers: list[LiteraturePaper] = []
        seen: set[str] = set()
        for paper_id in sorted(
            set(plan.seed_paper_ids).union(
                paper_id for query in plan.queries for paper_id in query.new_paper_ids
            )
        ):
            paper = self.kernel.papers.get(paper_id)
            if paper is not None and paper.paper_id not in seen:
                seen.add(paper.paper_id)
                papers.append(paper)
        return compute_coverage(plan, papers)


# --------------------------------------------------------------------------------------
# Record -> paper conversion
# --------------------------------------------------------------------------------------

#: Fields copied verbatim from a provider record. Deliberately metadata only: no interpretation.
_COPIED_FIELDS: tuple[str, ...] = (
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "url",
    "abstract",
    "citation_count",
    "open_access_pdf",
    "license",
)

_FALLBACK_SUFFIX: Mapping[QueryFamily, str] = {
    QueryFamily.EXACT_TERMINOLOGY: "exact terminology",
    QueryFamily.SYNONYMS: "alternative terminology",
    QueryFamily.MECHANISTIC_EQUIVALENTS: "mechanism-level equivalent",
    QueryFamily.FUNCTIONAL_EQUIVALENTS: "functionally equivalent approach",
    QueryFamily.HISTORICAL_TERMINOLOGY: "earlier name historical term",
    QueryFamily.NEIGHBORING_COMMUNITIES: "neighbouring community terminology",
    QueryFamily.RECENT_TERMINOLOGY: "recent work",
    QueryFamily.CITATION_EXPANSION: "citation-graph neighbours",
    QueryFamily.REFERENCE_EXPANSION: "reference-list expansion",
    QueryFamily.SEMANTIC_NEIGHBOR: "semantic neighbours",
    QueryFamily.AUTHOR_VENUE: "author and venue probe",
}


def _paper_from_record(record: ProviderRecord) -> LiteraturePaper:
    """Build an abstract-level paper. Never a mechanism claim — the model would reject it."""
    return LiteraturePaper(
        title=record.title,
        authors=list(record.authors),
        year=record.year,
        venue=record.venue,
        doi=record.doi,
        arxiv_id=record.arxiv_id,
        url=record.url,
        abstract=record.abstract,
        citation_count=record.citation_count,
        open_access_pdf=record.open_access_pdf,
        license=record.license,
        fulltext_status=(
            FulltextStatus.ABSTRACT_LEVEL_ONLY if record.abstract else FulltextStatus.UNKNOWN
        ),
        providers=[record.provider],
        provider_ids={record.provider.value: record.provider_id},
        retrieved_at=record.fetched_at,
        raw_record=dict(record.raw),
        imported_from=f"provider:{record.provider.value}",
        tags=[record.query] if record.query else [],
    )


def _merge_record(existing: LiteraturePaper, record: ProviderRecord) -> LiteraturePaper:
    """Fill gaps on a known paper from a second provider. Returns ``existing`` when nothing changed.

    Touches neither the identity fields nor ``fulltext_status``: which record we first based
    inclusion on is part of the provenance, and a search result is not allowed to change how much
    of a paper we claim to have read.
    """
    changes: dict[str, object] = {}
    for field in _COPIED_FIELDS:
        current = getattr(existing, field)
        incoming = getattr(record, field)
        if current in (None, "") and incoming not in (None, ""):
            changes[field] = incoming
    providers = list(existing.providers)
    if record.provider not in providers:
        providers.append(record.provider)
    if providers != list(existing.providers):
        changes["providers"] = sorted(providers, key=lambda kind: kind.value)
    provider_ids = dict(existing.provider_ids)
    if provider_ids.get(record.provider.value) != record.provider_id:
        provider_ids[record.provider.value] = record.provider_id
        changes["provider_ids"] = provider_ids
    if not changes:
        return existing
    return existing.with_updates(**changes)  # type: ignore[return-value]


def _seed_identifier(paper: LiteraturePaper) -> str:
    """The most stable handle a provider can resolve for a seed paper."""
    if paper.doi:
        return paper.doi
    if paper.arxiv_id:
        return paper.arxiv_id
    for key in sorted(paper.provider_ids):
        return paper.provider_ids[key]
    return paper.title


__all__ = [
    "ExpandWithLLM",
    "GRAPH_FAMILIES",
    "HISTORICAL_TERMS",
    "MAX_LLM_QUERIES",
    "MAX_QUERIES_PER_FAMILY",
    "MECHANISM_EQUIVALENTS",
    "FUNCTIONAL_EQUIVALENTS",
    "NEIGHBOUR_COMMUNITIES",
    "QueryPlanner",
    "RECENT_TERMS",
    "SYNONYM_LEXICON",
]
