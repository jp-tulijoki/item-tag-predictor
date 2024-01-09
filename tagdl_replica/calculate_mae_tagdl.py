import pandas as pd

test = pd.read_csv("movie_dataset_public_final/processed/10folds/test0.csv")
test = test[["tag", "targets", "movieId"]] 
test = test.rename(columns={"movieId" : "item_id"})

predictions = pd.read_csv("predictions0/tagdl_predictions0.csv")
predictions.predictions = predictions.predictions * 4 + 1

evaluation = pd.merge(test, predictions, on=["tag", "item_id"], how="left")
evaluation["error"] = evaluation.targets - evaluation.predictions
print(evaluation["error"].abs().sum() / len(evaluation))