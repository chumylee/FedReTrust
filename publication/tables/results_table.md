# Publication results

Mean percentages over rounds 5–48, then over completed repeats.

| Method | Metric | MNIST C | MNIST P | MNIST H | Fashion-MNIST C | Fashion-MNIST P | Fashion-MNIST H | EMNIST Balanced C | EMNIST Balanced P | EMNIST Balanced H | Smartphone HAR C | Smartphone HAR P | Smartphone HAR H |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RFA | Accuracy | **89.36** | **86.42** | 86.24 | **73.46** | **71.17** | 66.67 | **58.96** | **51.77** | 38.18 | 68.95 | 66.25 | 67.85 |
| RFA | Precision | **90.02** | **88.08** | 86.22 | **74.05** | **73.23** | **63.16** | 61.68 | 55.89 | 41.17 | 69.79 | 66.48 | 69.58 |
| RFA | Recall | **89.36** | **86.42** | 82.66 | **73.46** | **71.17** | 54.93 | **58.96** | **51.77** | 33.67 | 68.04 | 64.86 | 66.18 |
| FLTrust | Accuracy | 86.22 | 84.30 | 63.29 | 68.91 | 67.34 | 19.95 | 50.08 | 47.06 | 8.70 | **70.26** | **69.46** | 66.88 |
| FLTrust | Precision | 87.45 | 86.20 | 69.97 | 70.32 | 68.84 | 19.06 | 54.55 | 51.58 | 9.09 | **71.73** | **70.26** | 68.84 |
| FLTrust | Recall | 86.22 | 84.30 | 61.17 | 68.91 | 67.34 | 17.72 | 50.08 | 47.06 | 9.20 | **69.32** | **68.59** | 65.65 |
| Viceroy (adapted) | Accuracy | 41.81 | 22.65 | 30.88 | 46.15† | 24.02† | 38.71† | 6.18 | 4.95 | 4.73 | 30.59 | 17.44 | 27.27† |
| Viceroy (adapted) | Precision | 39.77 | 18.49 | 26.14 | 44.49† | 17.33† | 30.29† | 3.66 | 2.68 | 1.56 | 20.93 | 10.66 | 15.89† |
| Viceroy (adapted) | Recall | 41.81 | 22.65 | 26.36 | 46.15† | 24.02† | 30.86† | 6.18 | 4.95 | 3.75 | 29.39 | 17.07 | 25.49† |
| FedReTrust | Accuracy | 88.97 | 84.44 | **87.30** | 70.15 | 67.94 | **67.62** | 58.86 | 51.52 | **41.89** | 69.50 | 64.56 | **68.18** |
| FedReTrust | Precision | 89.71 | 87.20 | **87.19** | 71.43 | 70.44 | 62.59 | **61.75** | **56.18** | **44.49** | 70.35 | 64.56 | **69.82** |
| FedReTrust | Recall | 88.97 | 84.44 | **83.46** | 70.15 | 67.94 | **56.38** | 58.86 | 51.52 | **36.28** | 68.58 | 63.28 | **66.47** |

C: clean training; P: label poisoning; H: honest drift. Precision and recall use macro averaging.
† Four completed repeats; all other entries use five. Diverged jobs remain in the raw evidence.
Under clean training and poisoning, macro recall equals accuracy on the three image datasets because their test sets are class-balanced.
