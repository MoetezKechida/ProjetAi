# Model comparison (test set, all patients)

| Model | Type | Val macro-F1 | Accuracy | Macro precision | Macro recall | Macro-F1 | Critical P / R / F1 | AUROC Critical | AUROC ≥ High |
|---|---|---|---|---|---|---|---|---|---|
| Persistence (current NEWS2 level) | rule | – | 0.611 | 0.417 | 0.362 | 0.356 | 0.00 / 0.00 / 0.00 | – | – |
| LogisticRegression | ML | 0.365 | 0.495 | 0.370 | 0.416 | 0.368 | 0.07 / 0.23 / 0.11 | 0.729 | 0.806 |
| DecisionTree | ML | 0.403 | 0.541 | 0.401 | 0.454 | 0.402 | 0.13 / 0.26 / 0.18 | 0.712 | 0.813 |
| **RandomForest** | ML | 0.426 | 0.559 | 0.426 | 0.474 | 0.424 | 0.21 / 0.24 / 0.22 | 0.762 | 0.831 |
| SVM | ML | 0.360 | 0.625 | 0.528 | 0.365 | 0.364 | 0.42 / 0.02 / 0.03 | 0.744 | 0.819 |
| kNN | ML | 0.368 | 0.617 | 0.541 | 0.360 | 0.362 | 0.54 / 0.03 / 0.06 | 0.695 | 0.803 |
| **GRU** | DL | 0.494 | 0.646 | 0.533 | 0.475 | 0.490 | 0.44 / 0.18 / 0.25 | 0.785 | 0.850 |
