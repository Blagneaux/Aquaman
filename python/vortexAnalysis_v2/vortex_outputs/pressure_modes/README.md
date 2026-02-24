# Pressure Classifier Summary

This readme summarizes the **average test accuracy** for each dataset using the values written on the **last line** of each summary CSV.

Dataset order in those lines is:
- `nadia`, `thomas`, `boai`, `full`

## Average Test Accuracy by Architecture

| Architecture | Summary file | nadia | thomas | boai | full |
| --- | --- | --- | --- | --- | --- |
| Linear (modes + linear layer) | `classifier_results_summary.csv` | 0.689247 | 0.631884 | 0.756000 | 0.622667 |
| Nonlinear MLP (modes + ReLU) | `classifier_nonlinear_results_summary.csv` | 0.712903 | 0.656522 | 0.808000 | 0.660000 |
| CNN on modes (conv1d + linear) | `classifier_cnn_results_summary.csv` | 0.678495 | 0.634783 | 0.716000 | 0.608000 |
| Full CNN (raw signal conv1d + linear) | `classifier_fullcnn_results_summary.csv` | 0.695699 | 0.621739 | 0.778667 | 0.621333 |

## Notes
- All averages are taken directly from the last line of each summary CSV.
- If you regenerate the summaries, update this table by re-reading those last lines.
