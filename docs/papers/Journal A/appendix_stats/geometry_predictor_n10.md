# Geometry Predictor — n=10

## Pooled regression (ΔMRR_q ~ standardized geometry features)

| feature | β | CI_low | CI_high |
|---|---:|---:|---:|
| gold_d15_sf | -0.0111 | -0.3962 | +0.5485 |
| gold_d15_sp | +0.0106 | -0.5091 | +0.3393 |
| cross_gold_margin | +0.0565 | -0.2666 | +0.2752 |
| joint_margin | -0.0270 | -0.1151 | +0.0400 |
| tau_signal | +0.0068 | -0.0635 | +0.0643 |
| sf_d15 | -0.0215 | -0.4455 | +0.2924 |
| sp_d15 | -0.0222 | -0.1515 | +0.1640 |
| kappa | +0.0343 | -0.0629 | +0.1470 |
| R² | 0.242 | | |

## Per-dataset
### hotpotqa (n=10)
- R²=0.350; gold_d15_sf β=+0.2829 (CI [-0.8345,+2.4848]); joint_margin β=-0.2524
- Type A=1 C=8 A_joint=-0.09587717265108647 C_joint=-0.16727528271258085

### musique (n=10)
- R²=0.731; gold_d15_sf β=-0.4053 (CI [-1.3905,+0.1563]); joint_margin β=+0.2782
- Type A=0 C=8 A_joint=None C_joint=-0.03627250231895496

### scifact (n=10)
- R²=1.000; gold_d15_sf β=+0.0231 (CI [-0.0135,+0.0250]); joint_margin β=+0.0023
- Type A=0 C=8 A_joint=None C_joint=0.40756065178701917

### 2wikimultihopqa (n=10)
- R²=0.000; gold_d15_sf β=+0.0000 (CI [+0.0000,+0.0000]); joint_margin β=+0.0000
- Type A=0 C=10 A_joint=None C_joint=-0.2582191314536943
