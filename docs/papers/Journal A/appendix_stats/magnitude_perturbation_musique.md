#### musique: magnitude perturbation on REAL component scores (n=10 queries)

Signal X transformed, other signal fixed. Each cell: MRR | tau(fused vs orig-fused).


**Perturbed signal: SF**

| Condition | linear MRR / tau | rrf MRR / tau | combsum MRR / tau | combmnz MRR / tau | borda MRR / tau | zscore MRR / tau | minmax MRR / tau |
|---|---|---|---|---|---|---|---|
| orig | MRR=0.950 tau=+1.000 | MRR=0.861 tau=+1.000 | MRR=0.914 tau=+1.000 | MRR=0.903 tau=+1.000 | MRR=0.855 tau=+1.000 | MRR=0.950 tau=+1.000 | MRR=0.950 tau=+1.000 |
| x2 | MRR=0.950 tau=+1.000 | MRR=0.861 tau=+1.000 | MRR=0.805 tau=+0.867 | MRR=0.803 tau=+0.892 | MRR=0.855 tau=+1.000 | MRR=0.950 tau=+1.000 | MRR=0.950 tau=+1.000 |
| log1p | MRR=0.933 tau=+0.958 | MRR=0.861 tau=+1.000 | MRR=0.917 tau=+0.966 | MRR=0.903 tau=+0.975 | MRR=0.855 tau=+1.000 | MRR=0.950 tau=+0.982 | MRR=0.933 tau=+0.958 |
| pow05 | MRR=0.925 tau=+0.837 | MRR=0.861 tau=+1.000 | MRR=0.906 tau=+0.800 | MRR=0.903 tau=+0.832 | MRR=0.855 tau=+1.000 | MRR=0.950 tau=+0.944 | MRR=0.925 tau=+0.837 |
| rpr | MRR=0.925 tau=+0.834 | MRR=0.861 tau=+1.000 | MRR=0.905 tau=+0.803 | MRR=0.903 tau=+0.833 | MRR=0.855 tau=+1.000 | MRR=0.950 tau=+0.914 | MRR=0.925 tau=+0.831 |
| shufflescores | MRR=0.900 tau=+0.522 | MRR=0.397 tau=+0.509 | MRR=0.565 tau=+0.463 | MRR=0.523 tau=+0.594 | MRR=0.293 tau=+0.551 | MRR=0.950 tau=+0.578 | MRR=0.933 tau=+0.518 |

**Perturbed signal: SPLADE**

| Condition | linear MRR / tau | rrf MRR / tau | combsum MRR / tau | combmnz MRR / tau | borda MRR / tau | zscore MRR / tau | minmax MRR / tau |
|---|---|---|---|---|---|---|---|
| orig | MRR=0.805 tau=+1.000 | MRR=0.861 tau=+1.000 | MRR=0.914 tau=+1.000 | MRR=0.903 tau=+1.000 | MRR=0.855 tau=+1.000 | MRR=0.908 tau=+1.000 | MRR=0.805 tau=+1.000 |
| x2 | MRR=0.805 tau=+1.000 | MRR=0.861 tau=+1.000 | MRR=0.920 tau=+0.864 | MRR=0.904 tau=+0.899 | MRR=0.855 tau=+1.000 | MRR=0.908 tau=+1.000 | MRR=0.805 tau=+1.000 |
| log1p | MRR=0.805 tau=+0.976 | MRR=0.861 tau=+1.000 | MRR=0.805 tau=+0.971 | MRR=0.803 tau=+0.979 | MRR=0.855 tau=+1.000 | MRR=0.857 tau=+0.987 | MRR=0.805 tau=+0.976 |
| pow05 | MRR=0.805 tau=+0.954 | MRR=0.861 tau=+1.000 | MRR=0.857 tau=+0.885 | MRR=0.853 tau=+0.905 | MRR=0.855 tau=+1.000 | MRR=0.857 tau=+0.976 | MRR=0.805 tau=+0.954 |
| rpr | MRR=0.579 tau=+0.779 | MRR=0.861 tau=+1.000 | MRR=0.744 tau=+0.834 | MRR=0.703 tau=+0.866 | MRR=0.855 tau=+1.000 | MRR=0.563 tau=+0.909 | MRR=0.595 tau=+0.776 |
| shufflescores | MRR=0.481 tau=+0.702 | MRR=0.341 tau=+0.523 | MRR=0.239 tau=+0.740 | MRR=0.475 tau=+0.797 | MRR=0.261 tau=+0.551 | MRR=0.295 tau=+0.693 | MRR=0.458 tau=+0.715 |
