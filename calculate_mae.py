import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


tag_path = 'movie_dataset_public_final/raw/tags.json'
prediction_file = 'predictions_20.csv'
item_tag_scores_path = 'item_tag_scores_20_tags.csv'
    
def create_item_tag_scores_predictions(prediction_file):
    prediction_df = pd.read_csv(prediction_file)
    item_tag_prediction_df = pd.DataFrame(columns=['item_id', 'tag_scores'])
    item_ids = prediction_df['item_id'].unique().tolist()
    for item_id in item_ids:
        predictions_per_item = prediction_df[prediction_df['item_id'] == item_id].predictions
        predictions_array = np.array(json.loads(predictions_per_item.iloc[0]))
        for i in range(1, predictions_per_item.shape[0]):
            predictions_array = np.vstack((predictions_array, np.array(json.loads(predictions_per_item.iloc[i]))))
        if predictions_array.ndim == 2:
            predictions_array = np.mean(predictions_array, axis=0)  
        item_tag_prediction_df = item_tag_prediction_df.append({'item_id': item_id, 'tag_scores': predictions_array}, ignore_index=True)
    return item_tag_prediction_df     

def calculate_mae(item_tag_predictions, item_tag_scores_path):
    item_tag_scores = pd.read_csv(item_tag_scores_path)
    item_ids = item_tag_predictions['item_id'].tolist()
    maes_per_item = []
    for item_id in item_ids:
        prediction = item_tag_predictions[item_tag_predictions['item_id'] == item_id]['tag_scores'].squeeze()
        target = np.array(json.loads(item_tag_scores[item_tag_scores['item_id'] == item_id]['tag_scores'].iloc[0]))
        maes_per_item.append(mean_absolute_error(prediction,target))
    print(np.mean(maes_per_item) * 4)            
    
item_tag_predictions = create_item_tag_scores_predictions(prediction_file)
calculate_mae(item_tag_predictions, item_tag_scores_path)

    