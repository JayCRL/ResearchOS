# Statistical audit — 2024-11-04 (stale)

Auditor: self-review

n = 3 seeds per arm.

| arm | retention@1 mean | std |
|---|---|---|
| direct | 0.402 | 0.004 |
| shufwrite | 0.268 | 0.005 |

Unpaired t-test: t = 31.4, df = 4, p = 0.00001. Effect size d = 25.6.

Conclusion: direct writeback improves retention@1 by 13.4 points.

NOTE: this audit was performed before the data-order fix on 2024-11-07. The numbers above are
therefore not comparable with later runs and should not be cited.
