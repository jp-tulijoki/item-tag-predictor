import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

tenfold_path = 'movie_dataset_public_final/processed/10folds'

def calculate_mae(predictions, test, i):
    evaluation = pd.merge(test, predictions, on=["tag", "item_id"], how="left").dropna()
    evaluation["error"] = evaluation.targets - evaluation.predictions
    mae = mean_absolute_error(evaluation.targets, evaluation.predictions)
    assert mae == evaluation["error"].abs().sum() / len(evaluation)
    print(f"MAE for set {i}: {mae}")         

for i in range(1):
    df = pd.read_csv("predictions0/predictions0_.csv")
    df = df.groupby(["tag", "item_id"]).mean().reset_index()
    df["predictions"] = df["predictions"] * 4 + 1  
    test = pd.read_csv("movie_dataset_public_final/processed/10folds/test0.csv").rename(columns={"movieId": "item_id"})
    calculate_mae(df, test, i)