import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

tag_path = 'movie_dataset_public_final/raw/tags.json'
tenfold_path = 'movie_dataset_public_final/processed/10folds'

def get_tags():
    with open(tag_path) as file:
        tag_dict = {}
        for line in file:
            json_dict = json.loads(line)
            tag_dict[json_dict["id"]] = json_dict["tag"]  
        return tag_dict
    
def create_item_tag_scores_predictions(prediction_file):
    prediction_df = pd.read_csv(f"predictions/{prediction_file}")
    tag_id = int(prediction_file.split("_")[-1].split(".")[0])
    tag = tags[tag_id]
    item_tag_prediction_df = pd.DataFrame(columns=['item_id', 'tag_id', 'tag', 'predictions'])
    item_ids = prediction_df['item_id'].unique().tolist()
    for item_id in item_ids:
        predictions_per_item = prediction_df[prediction_df['item_id'] == item_id].predictions
        predictions_array = np.array(json.loads(predictions_per_item.iloc[0]))
        for i in range(1, predictions_per_item.shape[0]):
            predictions_array = np.vstack((predictions_array, np.array(json.loads(predictions_per_item.iloc[i]))))
        if predictions_array.ndim == 2:
            predictions_array = np.mean(predictions_array, axis=0)  
        item_tag_prediction_df = item_tag_prediction_df.append({'item_id': item_id, 'tag_id': tag_id, 'tag': tag, 'predictions': predictions_array.item()}, ignore_index=True)
    return item_tag_prediction_df     

def make_predictions_and_calculate_mae(combined_prediction_df, tenfold):
    predictions = []
    tagDL = pd.read_csv(f"{tenfold_path}/test{tenfold}.csv").rename(columns={"movieId": "item_id"})
    for row in tagDL.iterrows():
        tag = row[1]["tag"]
        item_id = row[1]["item_id"]
        df_row = combined_prediction_df[(combined_prediction_df["item_id"] == item_id) & (combined_prediction_df["tag"] == tag)]
        if not df_row.empty:
            predictions.append(round(df_row["predictions"].item() * 4 + 1, 2))
        else:
            raise LookupError(f"Prediction for item {item_id} and tag {tag} not found.")
    with open(f"processed_data/bert_predictions{tenfold}.txt", "w") as file:
        file.write("\n".join(map(str, predictions)))
    targets = tagDL["targets"].to_list()
    print(f"MAE for set {tenfold}: {mean_absolute_error(predictions, targets)}")                
        
prediction_files = sorted(os.listdir("predictions"))

with open ("score_list.txt", "r") as file:
    score_list = file.read().split("\n")[:-1]  

combined_prediction_df = pd.DataFrame(columns=['item_id', 'tag_id', 'predictions'])
tags = get_tags()

for i, prediction_file in enumerate(prediction_files):
    score_file = score_list[i]
    target = pd.read_csv(f"scores/{score_file}")
    item_tag_predictions = create_item_tag_scores_predictions(prediction_file)
    combined_prediction_df = pd.concat([combined_prediction_df, item_tag_predictions], ignore_index=True)
    combined_prediction_df["item_id"].astype('Int64')

combined_prediction_df.to_csv("processed_data/merged_predictions/merged_predictions.csv", index=False)

for i in range(10):
    make_predictions_and_calculate_mae(combined_prediction_df, i)
