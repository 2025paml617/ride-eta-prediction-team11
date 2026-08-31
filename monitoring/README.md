# Drift simulation and retraining trigger

Run the deterministic drift simulation with:

```powershell
python monitoring/drift_simulation.py
```

Outputs:

- `monitoring/drift_metrics.csv`: PSI and relative mean shift per feature
- `monitoring/drift_report.json`: summary and retraining decision

The default trigger is PSI `>= 0.20` for any monitored feature. A triggered
run means the current feature distribution is materially different from the
reference distribution and should start the retraining workflow:

```powershell
python monitoring/drift_simulation.py --fail-on-drift
python preprocessing/preprocessing.py
python features/build_features.py
python models/train_linear_regression.py
python models/train_advanced_models.py
python models/evaluate_and_select_best.py
```

The simulation is reproducible with `--seed`, and its intensity can be varied
with `--drift-scale` for demonstration and testing.
