import mlflow

# Fetch experiment details
experiment = mlflow.get_experiment_by_name("NYC_Taxi_Trip_Duration")

# Query all logged runs into a Pandas DataFrame
runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# View key columns
print(runs_df[["run_id", "tags.mlflow.runName", "metrics.RMSE", "metrics.MAE", "metrics.R2"]])