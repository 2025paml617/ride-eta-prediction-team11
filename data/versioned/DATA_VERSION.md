# Dataset versioning

The raw and processed datasets are pipeline inputs/outputs managed with DVC.
The current checked-in version is recorded in `manifest.json`; it includes
content hashes and schemas for both the taxi and weather inputs.

Run the following after replacing either source file:

```powershell
dvc add data/raw/NYC.csv
dvc add data/raw/weather.csv
dvc repro
dvc status
dvc push
```

Commit the generated `.dvc` file, `dvc.lock`, and the validation report for
each dataset revision. Configure a shared DVC remote before pushing so another
reviewer can reproduce the exact dataset and model inputs from a clean clone.

For an independently reviewable content hash, also generate a manifest:

```powershell
python scripts/version_dataset.py data/raw/NYC.csv data/versioned/manifest.json
```

Commit the manifest with the corresponding DVC revision.
