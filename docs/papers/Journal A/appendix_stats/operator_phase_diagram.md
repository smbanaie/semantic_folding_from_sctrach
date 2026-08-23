# Synthetic Operator Phase Diagram (Item 15)

30 trials/cell; N in {20,100,500}; families shape signal A; signal B carries a magnitude spike on the gold doc in magnitude-relevant regimes; B scaled x{1,10,100}; deterministic seeds.

## Mean top-1 accuracy (averaged over pool sizes and scales)

| family | regime | linear | rrf | combsum | combmnz | borda | zscore | minmax | winner |
|--------|--------|------:|------:|------:|------:|------:|------:|------:|--------|
| concentrated | rank-dominant | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **linear** *(all tie)* |
| concentrated | magnitude-dominant | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | **linear** |
| concentrated | mixed | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | **linear** |
| spread | rank-dominant | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **linear** *(all tie)* |
| spread | magnitude-dominant | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | **linear** |
| spread | mixed | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | **linear** |
| heavy-tail | rank-dominant | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **linear** *(all tie)* |
| heavy-tail | magnitude-dominant | 1.000 | 0.000 | 0.996 | 0.996 | 0.000 | 1.000 | 1.000 | **linear** |
| heavy-tail | mixed | 1.000 | 0.000 | 0.985 | 0.985 | 0.000 | 1.000 | 1.000 | **linear** |
