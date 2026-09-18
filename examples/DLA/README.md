# DLA — Direct vs Localized Assignment in fast-weight memory

Informal working repo for a study on where fast weights are written back.

## Research question

Does *where* a fast weight is written back to (its originating position vs a shuffled position)
determine how much of it survives into the next context window?

## Current claims

- Direct writeback retains more than shuffled writeback at the 124M scale (matched control, 3 seeds).
- Retention differences track the *alignment* between write position and read position, not the
  number of parameters updated.
- Sleep-style consolidation is necessary for retention beyond one window.

## Rejected claims

- ~~The effect is an optimisation artefact: we rejected this after matching optimizer and LR schedule.~~
- ~~Shuffled writeback is simply noisier: rejected, variance is equal across arms.~~

## Headline numbers (see runs/metrics.csv)

| arm | retention@1 | retention@4 | seeds |
|---|---|---|---|
| direct | 0.418 | 0.377 | 0,1,2 |
| shufwrite | 0.271 | 0.199 | 0,1,2 |

Direct beats shuffled writeback by 14.7 points at retention@1. This establishes the mechanism.

## Open questions

- Does the effect survive at 1B scale?
- Is the matched-energy control actually matched on FLOPs, or only on steps?

## Notes

This README predates the last two experiment runs and may be stale.
