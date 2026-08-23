#### hotpotqa: magnitude perturbation on REAL component scores (n=10 queries)

Signal X transformed, other signal fixed. Each cell: MRR | tau(fused vs orig-fused).


**Perturbed signal: SF**

| Condition | linear MRR / tau | rrf MRR / tau | combsum MRR / tau | combmnz MRR / tau | borda MRR / tau | zscore MRR / tau | minmax MRR / tau |
|---|---|---|---|---|---|---|---|
| orig | MRR=1.000 tau=+1.000 | MRR=0.883 tau=+1.000 | MRR=1.000 tau=+1.000 | MRR=1.000 tau=+1.000 | MRR=0.733 tau=+1.000 | MRR=1.000 tau=+1.000 | MRR=1.000 tau=+1.000 |
| x2 | MRR=1.000 tau=+1.000 | MRR=0.883 tau=+1.000 | MRR=0.867 tau=+0.933 | MRR=0.867 tau=+0.933 | MRR=0.733 tau=+1.000 | MRR=1.000 tau=+1.000 | MRR=1.000 tau=+1.000 |
| log1p | MRR=1.000 tau=+0.975 | MRR=0.883 tau=+1.000 | MRR=1.000 tau=+0.997 | MRR=1.000 tau=+0.997 | MRR=0.733 tau=+1.000 | MRR=1.000 tau=+0.981 | MRR=1.000 tau=+0.975 |
| pow05 | MRR=1.000 tau=+0.853 | MRR=0.883 tau=+1.000 | MRR=1.000 tau=+0.821 | MRR=1.000 tau=+0.821 | MRR=0.733 tau=+1.000 | MRR=1.000 tau=+0.867 | MRR=1.000 tau=+0.853 |
| rpr | MRR=1.000 tau=+0.689 | MRR=0.883 tau=+0.993 | MRR=1.000 tau=+0.687 | MRR=1.000 tau=+0.690 | MRR=0.733 tau=+0.989 | MRR=1.000 tau=+0.730 | MRR=1.000 tau=+0.687 |
| shufflescores | MRR=0.883 tau=+0.653 | MRR=0.354 tau=+0.427 | MRR=0.520 tau=+0.552 | MRR=0.587 tau=+0.546 | MRR=0.219 tau=+0.437 | MRR=0.900 tau=+0.682 | MRR=0.850 tau=+0.654 |

**Perturbed signal: SPLADE**

| Condition | linear MRR / tau | rrf MRR / tau | combsum MRR / tau | combmnz MRR / tau | borda MRR / tau | zscore MRR / tau | minmax MRR / tau |
|---|---|---|---|---|---|---|---|
| orig | MRR=0.867 tau=+1.000 | MRR=0.883 tau=+1.000 | MRR=1.000 tau=+1.000 | MRR=1.000 tau=+1.000 | MRR=0.733 tau=+1.000 | MRR=0.867 tau=+1.000 | MRR=0.867 tau=+1.000 |
| x2 | MRR=0.867 tau=+1.000 | MRR=0.883 tau=+1.000 | MRR=1.000 tau=+0.942 | MRR=1.000 tau=+0.942 | MRR=0.733 tau=+1.000 | MRR=0.867 tau=+1.000 | MRR=0.867 tau=+1.000 |
| log1p | MRR=0.783 tau=+0.969 | MRR=0.883 tau=+1.000 | MRR=0.933 tau=+0.994 | MRR=0.933 tau=+0.994 | MRR=0.733 tau=+1.000 | MRR=0.783 tau=+0.977 | MRR=0.783 tau=+0.969 |
| pow05 | MRR=0.617 tau=+0.887 | MRR=0.883 tau=+1.000 | MRR=0.933 tau=+0.908 | MRR=0.933 tau=+0.908 | MRR=0.733 tau=+1.000 | MRR=0.617 tau=+0.896 | MRR=0.617 tau=+0.887 |
| rpr | MRR=0.470 tau=+0.799 | MRR=0.883 tau=+0.986 | MRR=0.537 tau=+0.837 | MRR=0.537 tau=+0.840 | MRR=0.733 tau=+0.981 | MRR=0.462 tau=+0.825 | MRR=0.475 tau=+0.800 |
| shufflescores | MRR=0.523 tau=+0.476 | MRR=0.489 tau=+0.440 | MRR=0.488 tau=+0.325 | MRR=0.454 tau=+0.367 | MRR=0.210 tau=+0.418 | MRR=0.498 tau=+0.472 | MRR=0.448 tau=+0.461 |
