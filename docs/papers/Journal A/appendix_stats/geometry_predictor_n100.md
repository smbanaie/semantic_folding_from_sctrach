# Geometry Predictor — n=100

## Pooled regression (ΔMRR_q ~ standardized geometry features)

| feature | β | CI_low | CI_high |
|---|---:|---:|---:|
| gold_d15_sf | -0.0757 | -0.1556 | +0.0055 |
| gold_d15_sp | +0.0900 | +0.0102 | +0.1766 |
| cross_gold_margin | +0.0673 | -0.0075 | +0.1415 |
| joint_margin | -0.0103 | -0.0724 | +0.0536 |
| tau_signal | +0.0052 | -0.0180 | +0.0291 |
| sf_d15 | +0.0100 | -0.0294 | +0.0503 |
| sp_d15 | -0.0528 | -0.1188 | +0.0099 |
| kappa | -0.0180 | -0.0385 | -0.0005 |
| R² | 0.045 | | |

## Per-dataset
### hotpotqa (n=100)
- R²=0.133; gold_d15_sf β=+0.0153 (CI [-0.1986,+0.0857]); joint_margin β=-0.0108
- Type A=4 C=75 A_joint=-0.2295095854708596 C_joint=-0.09814657655737076

### musique (n=100)
- R²=0.191; gold_d15_sf β=-0.1177 (CI [-0.2350,-0.0034]); joint_margin β=+0.0397
- Type A=5 C=71 A_joint=-0.09998884252335855 C_joint=-0.08362583702985936

### nq_rear (n=100)
- R²=0.104; gold_d15_sf β=+0.1811 (CI [-0.1935,+0.3472]); joint_margin β=+0.0855
- Type A=3 C=80 A_joint=-0.11173972714857965 C_joint=-0.06955010321044101
