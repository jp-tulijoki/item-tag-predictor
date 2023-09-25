import json
import pandas as pd
import random
import os
import warnings
warnings.filterwarnings('ignore')

review_path = 'movie_dataset_public_final/raw/reviews.json'
tag_path = 'movie_dataset_public_final/raw/tags.json'
survey_path = 'movie_dataset_public_final/raw/survey_answers.json'

def get_reviews(score_df):
    with open(review_path) as file:
        reviews = []
        reviews_per_item_train = {}
        reviews_per_item_test = {}
        items_to_take = score_df["item_id"].to_list()
        for line in file:
            json_dict = json.loads(line)
            id = json_dict["item_id"]
            if id not in items_to_take:
                continue
            json_dict["txt"] = json_dict["txt"].replace('\r', '').replace('\t', ' ')
            reviews.append(json_dict)
        random.shuffle(reviews)    
        sampled_train = []
        sampled_test = []               
        for review in reviews:
            id = review["item_id"]
            if id in reviews_per_item_test.keys():
                if reviews_per_item_train[id] < 50:
                    if reviews_per_item_test[id] * 9 < reviews_per_item_train[id]:
                        reviews_per_item_test[id] = reviews_per_item_test[id] + 1
                        sampled_test.append(review)
                    else:
                        reviews_per_item_train[id] = reviews_per_item_train[id] + 1
                        sampled_train.append(review)    
            else:
                reviews_per_item_test[id] = 1
                reviews_per_item_train[id] = 0
                sampled_test.append(review)
        train_reviews = pd.DataFrame.from_records(sampled_train)
        test_reviews = pd.DataFrame.from_records(sampled_test)        
        return train_reviews, test_reviews       

def get_tags():
    with open(tag_path) as file:
        tag_dict = {}
        tag_index = []
        for line in file:
            json_dict = json.loads(line)
            tag_dict[json_dict["tag"]] = json_dict["id"]
            tag_index.append(json_dict["id"])      
        return tag_dict, tag_index
    
def get_selected_items(TAG_NO):
    with open(survey_path) as file:
        selected_items = []
    with open(survey_path) as file:
        for line in file:
            answer_dict = json.loads(line)
            item_id = answer_dict["item_id"]
            tag_id = answer_dict["tag_id"]
            if tag_id == TAG_NO and item_id not in selected_items:
                selected_items.append(item_id)
    return selected_items                          

def get_item_tag_scores(TAG_NO, TAG_ID):
    item_tag_dict = {}
    with open(survey_path) as file:
        for line in file:
            answer_dict = json.loads(line)
            item_id = answer_dict["item_id"]
            score = answer_dict["score"]
            tag_id = answer_dict["tag_id"]
            if score != -1 and tag_id == TAG_ID:
                if item_id not in item_tag_dict:
                    item_tag_dict[item_id] = {tag_id: {"cumulative_score": score, "scores_total": 1}}
                    continue
                tag_dict = item_tag_dict[item_id]
                if tag_id not in tag_dict:
                    tag_dict[tag_id] = {"cumulative_score": score, "scores_total": 1}
                else:
                    tag_dict[tag_id] = {"cumulative_score": score + tag_dict[tag_id]["cumulative_score"], "scores_total": tag_dict[tag_id]["scores_total"] + 1}  
    df = pd.DataFrame(columns=['item_id', 'tag_score'])
    for item in item_tag_dict:
        for tag in item_tag_dict[item]:
            scores = item_tag_dict[item][tag]["cumulative_score"]
            total = item_tag_dict[item][tag]["scores_total"]
            tag_score = (scores - total) / (total * 4) 
        df = df.append({'item_id': item, 'tag_score': tag_score}, ignore_index=True)
    df["item_id"] = df["item_id"].astype(int)        
    df.to_csv(f'scores/item_tag_scores_no_{TAG_NO}_id_{TAG_ID}.csv', index=False)                
    return df

tags, tag_index = get_tags()

for i in range(len(tag_index)):
    print(f"processing tag no {i} with id {tag_index[i]}")
    score_df = get_item_tag_scores(i, tag_index[i]) 
    if score_df.empty:
        continue   
    train_reviews, test_reviews = get_reviews(score_df)
    train_data = pd.merge(train_reviews, score_df, left_on='item_id', right_on='item_id', how='left')
    test_data = pd.merge(test_reviews, score_df, left_on='item_id', right_on='item_id', how='left')
    train_data.to_csv(f'reviews/train_data_tag_no_{i}_id_{tag_index[i]}.csv', index=False)
    test_data.to_csv(f'reviews/test_data_tag_no_{i}_id_{tag_index[i]}.csv', index=False)

review_list = sorted(list(filter(lambda r: "test" in r, os.listdir("reviews"))))

score_list = sorted(os.listdir("scores"))

with open("review_list.txt", "w") as file:
    for review in review_list:
        file.write(f"{review}\n")
        
with open("score_list.txt", "w") as file:
    for score in score_list:
        file.write(f"{score}\n")         
