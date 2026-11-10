# Example held-out results (20 synthetic seeds)

Run: `python run_experiment.py --output results --test-seeds 20`.

| Policy | Mean QoE surrogate | Mean stall seconds per video segment | Critical slice delivered fraction | Mean aggregate reward |
|---|---:|---:|---:|---:|
| Static reservation | 0.727 | 0.000 | 0.888 | 1.215 |
| Demand-proportional candidate | 0.721 | 0.000 | 0.890 | 1.211 |
| Learned contextual allocation | 0.730 | 0.004 | 0.957 | 1.255 |

All results depend on the chosen illustrative reward weights, traffic model, and random seeds; they are not an independent field measurement. In particular, the learned controller prioritizes the critical slice more heavily and exhibits a small mean video-stall cost. The static and proportional baselines achieve virtually zero average stalls in this experiment due to initial buffers and conservative ABR. Change outage duration / bandwidth and reward weights for sensitivity analysis rather than claiming universal dominance.

**One-step capacity forecasting:** Ridge MAE = 1.346 Mbps, persistence MAE = 1.348 Mbps. The improvement is small; it should not be represented as a substantial accuracy gain.
