import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

tag_path = 'movie_dataset_public_final/raw/tags.json'
tenfold_path = 'movie_dataset_public_final/processed/10folds'
PREDICTION_DF_EXISTS = False

def get_tags():
    with open(tag_path) as file:
        tag_dict = {}
        for line in file:
            json_dict = json.loads(line)
            tag_dict[json_dict["id"]] = json_dict["tag"]  
        return tag_dict
    
def create_item_tag_scores_predictions(prediction_file, tags):
    prediction_df = pd.read_csv(f"predictions/{prediction_file}")
    tag_id = int(prediction_file.split("_")[-1].split(".")[0])
    tag = tags[tag_id]
    item_tag_prediction_df = pd.DataFrame(columns=['item_id', 'tag_id', 'tag', 'predictions'])
    item_ids = prediction_df['item_id'].unique().tolist()
    for item_id in item_ids:
        predictions_per_item = prediction_df[prediction_df['item_id'] == item_id].predictions
        mean_prediction = np.mean(list(map(lambda x: json.loads(x)[0], predictions_per_item.to_list())))   
        item_tag_prediction_df = item_tag_prediction_df.append({'item_id': item_id, 'tag_id': tag_id, 'tag': tag, 'predictions': mean_prediction}, ignore_index=True)
    return item_tag_prediction_df     

def make_predictions_for_tenfolds(combined_prediction_df, tenfold):
    test = pd.read_csv(f"{tenfold_path}/test{tenfold}.csv").rename(columns={"movieId": "item_id"})
    predictions = test[["tag", "item_id"]].copy()
    predictions = pd.merge(predictions, combined_prediction_df, on=["tag", "item_id"], how="left")
    predictions["predictions"] = round(predictions["predictions"] * 4 + 1, 2)
    predictions.to_csv(f"processed_data/bert_predictions{tenfold}.csv", index=False)
    return predictions, test

def calculate_mae(predictions, test, i):
    evaluation = pd.merge(test, predictions, on=["tag", "item_id"], how="left")
    evaluation["error"] = evaluation.targets - evaluation.predictions
    mae = mean_absolute_error(evaluation.targets, evaluation.predictions)
    assert mae == evaluation["error"].abs().sum() / len(evaluation)
    print(f"MAE for set {i}: {mae}")           

def make_combined_prediction_df():
    prediction_files = sorted(os.listdir("predictions"))
    combined_prediction_df = pd.DataFrame(columns=['item_id', 'tag_id', 'predictions'])
    tags = get_tags()

    for prediction_file in prediction_files:
        item_tag_predictions = create_item_tag_scores_predictions(prediction_file, tags)
        combined_prediction_df = pd.concat([combined_prediction_df, item_tag_predictions], ignore_index=True)
        combined_prediction_df["item_id"].astype('Int64')
    combined_prediction_df.to_csv("processed_data/merged_predictions/merged_predictions.csv", index=False)

if not PREDICTION_DF_EXISTS:
    make_combined_prediction_df()

combined_prediction_df = pd.read_csv("processed_data/merged_predictions/merged_predictions.csv")

for i in range(10):
    predictions, test = make_predictions_for_tenfolds(combined_prediction_df, i)
    calculate_mae(predictions, test, i)

