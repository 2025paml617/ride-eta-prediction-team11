# Raw data contract

Place the source taxi CSV at `data/raw/NYC.csv`. The ingestion stage validates
the required taxi schema and records row-level rejection counts in
`data/processed/validation_report.json`.

Weather enrichment is optional. To enable it, place `data/raw/weather.csv` here.
The pipeline accepts either the normalized columns below or the supplied NYC
weather export, which uses `pickup_datetime`, `tempm`, `precipm`, `wspdm`, and
`vism`.

```text
weather_datetime,temperature_c,precipitation_mm,wind_speed_kmh,visibility_km
```

The weather timestamp is rounded to the hour and joined to pickup time. Missing
weather observations are retained as missing values and imputed during model
preprocessing.
