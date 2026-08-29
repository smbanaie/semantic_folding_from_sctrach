# Geometry Predictor — n=100

## Pooled regression (ΔMRR_q ~ standardized geometry features)

| feature | β | CI_low | CI_high |
|---|---:|---:|---:|
| gold_d15_sf | +0.0919 | -0.1440 | +3.1162 |
| gold_d15_sp | -0.2366 | -3.8758 | +0.0389 |
| cross_gold_margin | -0.1767 | -3.8508 | +0.0418 |
| joint_margin | +0.0513 | -0.0980 | +0.2228 |
| tau_signal | -0.0014 | -0.0476 | +0.0484 |
| sf_d15 | -0.0753 | -1.2488 | +0.0145 |
| sp_d15 | +0.1730 | -0.0083 | +2.8749 |
| kappa | -0.0323 | -0.0865 | +0.0209 |
| R² | 0.088 | | |

## Per-dataset
### hotpotqa (n=100)
- R²=0.088; gold_d15_sf β=+0.0919 (CI [-0.1440,+3.1162]); joint_margin β=+0.0513
- Type A=3 C=75 A_joint=-0.29693989644323254 C_joint=-0.09316546817348713
