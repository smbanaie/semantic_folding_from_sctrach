#### scifact: magnitude perturbation on REAL component scores (n=10 queries)

Signal X transformed, other signal fixed. Each cell: MRR | tau(fused vs orig-fused).


**Perturbed signal: SF**

| Condition | linear MRR / tau | rrf MRR / tau | combsum MRR / tau | combmnz MRR / tau | borda MRR / tau | zscore MRR / tau | minmax MRR / tau |
|---|---|---|---|---|---|---|---|
| orig | MRR=0.823 tau=+1.000 | MRR=0.821 tau=+1.000 | MRR=0.820 tau=+1.000 | MRR=0.820 tau=+1.000 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+1.000 | MRR=0.823 tau=+1.000 |
| x2 | MRR=0.823 tau=+1.000 | MRR=0.821 tau=+1.000 | MRR=0.818 tau=+0.913 | MRR=0.818 tau=+0.913 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+1.000 | MRR=0.823 tau=+1.000 |
| log1p | MRR=0.823 tau=+0.970 | MRR=0.821 tau=+1.000 | MRR=0.824 tau=+0.990 | MRR=0.824 tau=+0.990 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+0.977 | MRR=0.823 tau=+0.970 |
| pow05 | MRR=0.824 tau=+0.901 | MRR=0.821 tau=+1.000 | MRR=0.822 tau=+0.865 | MRR=0.822 tau=+0.865 | MRR=0.821 tau=+1.000 | MRR=0.824 tau=+0.910 | MRR=0.824 tau=+0.901 |
| rpr | MRR=0.823 tau=+0.733 | MRR=0.821 tau=+1.000 | MRR=0.826 tau=+0.711 | MRR=0.827 tau=+0.712 | MRR=0.821 tau=+1.000 | MRR=0.824 tau=+0.819 | MRR=0.825 tau=+0.735 |
| shufflescores | MRR=0.817 tau=+0.694 | MRR=0.206 tau=+0.434 | MRR=0.516 tau=+0.543 | MRR=0.425 tau=+0.525 | MRR=0.290 tau=+0.428 | MRR=0.819 tau=+0.686 | MRR=0.819 tau=+0.696 |
| compress | MRR=0.825 tau=+0.728 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+0.767 | MRR=0.823 tau=+0.767 | MRR=0.821 tau=+1.000 | MRR=0.824 tau=+0.817 | MRR=0.825 tau=+0.728 |
| amplify | MRR=0.823 tau=+0.872 | MRR=0.821 tau=+1.000 | MRR=0.719 tau=+0.782 | MRR=0.719 tau=+0.782 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+0.878 | MRR=0.823 tau=+0.872 |
| magswap | MRR=0.823 tau=+0.981 | MRR=0.821 tau=+1.000 | MRR=0.820 tau=+1.000 | MRR=0.820 tau=+1.000 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+0.994 | MRR=0.823 tau=+0.981 |

**Perturbed signal: SPLADE**

| Condition | linear MRR / tau | rrf MRR / tau | combsum MRR / tau | combmnz MRR / tau | borda MRR / tau | zscore MRR / tau | minmax MRR / tau |
|---|---|---|---|---|---|---|---|
| orig | MRR=0.818 tau=+1.000 | MRR=0.821 tau=+1.000 | MRR=0.820 tau=+1.000 | MRR=0.820 tau=+1.000 | MRR=0.821 tau=+1.000 | MRR=0.818 tau=+1.000 | MRR=0.818 tau=+1.000 |
| x2 | MRR=0.818 tau=+1.000 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+0.920 | MRR=0.823 tau=+0.920 | MRR=0.821 tau=+1.000 | MRR=0.818 tau=+1.000 | MRR=0.818 tau=+1.000 |
| log1p | MRR=0.767 tau=+0.969 | MRR=0.821 tau=+1.000 | MRR=0.820 tau=+0.987 | MRR=0.820 tau=+0.987 | MRR=0.821 tau=+1.000 | MRR=0.818 tau=+0.978 | MRR=0.767 tau=+0.969 |
| pow05 | MRR=0.719 tau=+0.929 | MRR=0.821 tau=+1.000 | MRR=0.820 tau=+0.930 | MRR=0.820 tau=+0.930 | MRR=0.821 tau=+1.000 | MRR=0.719 tau=+0.935 | MRR=0.719 tau=+0.929 |
| rpr | MRR=0.697 tau=+0.834 | MRR=0.821 tau=+1.000 | MRR=0.715 tau=+0.864 | MRR=0.766 tau=+0.864 | MRR=0.821 tau=+1.000 | MRR=0.698 tau=+0.902 | MRR=0.715 tau=+0.844 |
| shufflescores | MRR=0.823 tau=+0.492 | MRR=0.627 tau=+0.451 | MRR=0.503 tau=+0.352 | MRR=0.613 tau=+0.328 | MRR=0.297 tau=+0.415 | MRR=0.677 tau=+0.461 | MRR=0.655 tau=+0.496 |
| compress | MRR=0.696 tau=+0.835 | MRR=0.821 tau=+1.000 | MRR=0.704 tau=+0.588 | MRR=0.704 tau=+0.588 | MRR=0.821 tau=+1.000 | MRR=0.697 tau=+0.896 | MRR=0.696 tau=+0.835 |
| amplify | MRR=0.819 tau=+0.725 | MRR=0.821 tau=+1.000 | MRR=0.823 tau=+0.643 | MRR=0.823 tau=+0.643 | MRR=0.821 tau=+1.000 | MRR=0.818 tau=+0.725 | MRR=0.819 tau=+0.725 |
| magswap | MRR=0.767 tau=+0.972 | MRR=0.821 tau=+1.000 | MRR=0.820 tau=+0.999 | MRR=0.820 tau=+0.999 | MRR=0.821 tau=+1.000 | MRR=0.818 tau=+0.992 | MRR=0.767 tau=+0.972 |
