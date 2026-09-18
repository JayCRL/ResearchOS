# Research diary — DLA

## 2024-11-03

Ran the first direct-vs-shufwrite comparison. Direct looked better, 0.402 vs 0.268. I assumed this
was just extra capacity and did not believe it.

## 2024-11-07

Anomaly: the gap vanished when I trained with a different data order. Realised the two arms were not
seeing the same token order — the shufwrite arm was effectively getting a different curriculum.
Fixed the data ordering so both arms see identical batches. This changed the result substantially,
so the earlier numbers are not comparable.

## 2024-11-11

We decided to add a matched-energy control where the shuffled arm performs the same number of
parameter updates as the direct arm. Without it, a reviewer will say we are measuring capacity, not
placement.

We also decided to drop the 1B run for now: the compute budget is not there this quarter.

## 2024-11-14

Hypothesis revision: I originally thought retention came from the number of parameters updated. The
matched-parameter run shows that is not true — with the same number of parameters updated, the direct
arm still wins by 0.12. So the relevant variable looks like position alignment rather than capacity.

## 2024-11-18

Tried removing the sleep/consolidation step (`sleep_steps: 0`). Retention collapsed to chance beyond
one window. This is the first evidence that consolidation is *necessary*, not just helpful.

## 2024-11-21

Rejected claim: "direct writeback is a better optimiser". The loss curves are indistinguishable once
LR and schedule are matched, so the effect is not optimisation.

Open question: does the alignment effect hold for read positions that never appear in the write
window? I have no clean experiment for this yet.

TODO: re-run seed 3, it crashed halfway (OOM) and never finished. Do not report partial numbers.
