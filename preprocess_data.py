import json
import pandas as pd
import random

review_path = 'movie_dataset_public_final/raw/reviews.json'
tag_path = 'movie_dataset_public_final/raw/tags.json'
survey_path = 'movie_dataset_public_final/raw/survey_answers.json'
tag_count_path = 'movie_dataset_public_final/raw/tag_count.json'
TAGS_SELECTED = 25
MIN_REVIEWS = 100
SAMPLE_SIZE = 0.1

def get_reviews_for_tenfolds(item_tag_dict):

    with open(review_path) as file:
        reviews = []
        reviews_per_item = {}
        items_to_take = item_tag_dict.keys()
        for i, line in enumerate(file):
            json_dict = json.loads(line)
            id = json_dict["item_id"]
            if id not in items_to_take:
                continue
            json_dict["txt"] = json_dict["txt"].replace('\r', '').replace('\t', ' ')
            reviews.append(json_dict)
        sampled = random.sample(reviews, int(len(reviews) * SAMPLE_SIZE))                
        for review in sampled:
            id = review["item_id"]
            if id in reviews_per_item.keys():
                reviews_per_item[id] = reviews_per_item[id] + 1
            else:
                reviews_per_item[id] = 1                     
        review_df = pd.DataFrame.from_records(sampled)
        sorted_reviews_per_item = sorted(reviews_per_item.items(), key=lambda x: x[1], reverse=True)
        sorted_reviews_per_item = [item for item in sorted_reviews_per_item if item[1] >= MIN_REVIEWS]
        item_ids = [list() for i in range(10)]
        no_items = [0] * 10
        for item in sorted_reviews_per_item:
            item_id = item[0]
            no_reviews = item[1]        
            least_items_index = no_items.index(min(no_items))
            item_ids[least_items_index].append(item_id)
            no_items[least_items_index] += no_reviews
        return review_df, item_ids        

def get_tag_indexes(selected_tags):
    with open(tag_path) as file:
        tag_dict = {}
        tag_index = []
        i = 0
        for line in file:
            json_dict = json.loads(line)
            tag_id = json_dict["id"]
            if tag_id in selected_tags:
                tag_dict[tag_id] = i
                tag_index.append(tag_id)
                i += 1            
        return tag_dict
    
def select_tags_and_items():
    with open(survey_path) as file:
        tag_count = {}
        selected_items = []
        for line in file:
            answer_dict = json.loads(line)
            score = answer_dict["score"]
            tag_id = answer_dict["tag_id"]
            if score != 1:
                if tag_id not in tag_count:
                    tag_count[tag_id] = 1
                else:
                    tag_count[tag_id] += 1
        sorted_tag_count = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)  
        selected_tags = [item[0] for item in sorted_tag_count[:TAGS_SELECTED]]
    with open(survey_path) as file:
        for line in file:
            answer_dict = json.loads(line)
            item_id = answer_dict["item_id"]
            tag_id = answer_dict["tag_id"]
            if tag_id in selected_tags and item_id not in selected_items:
                selected_items.append(item_id)
    return selected_tags, selected_items        
           
def get_tag_applications_per_item(selected_tags, selected_items):
    with open(tag_count_path) as file:
        item_tag_application_dict = {}
        for line in file:
            tag_count_dict = json.loads(line)
            item_id = tag_count_dict["item_id"]
            tag_id = tag_count_dict["tag_id"]
            if tag_id not in selected_tags or item_id not in selected_items:
                continue
            if item_id not in item_tag_application_dict:
                item_tag_application_dict[item_id] = []
            if tag_id not in item_tag_application_dict[item_id]:
                item_tag_application_dict[item_id].append(tag_id)
        return item_tag_application_dict            


def get_item_tag_scores(selected_tags, selected_items):
    with open(survey_path) as file:
        item_tag_dict = {}
        for line in file:
            answer_dict = json.loads(line)
            item_id = answer_dict["item_id"]
            score = answer_dict["score"]
            tag_id = answer_dict["tag_id"] 
            if score != -1 and tag_id in selected_tags and item_id in selected_items:
                if item_id not in item_tag_dict:
                    item_tag_dict[item_id] = {tag_id: {"cumulative_score": score, "scores_total": 1}}
                    continue
                tag_dict = item_tag_dict[item_id]
                if tag_id not in tag_dict:
                    tag_dict[tag_id] = {"cumulative_score": score, "scores_total": 1}
                else:
                    tag_dict[tag_id] = {"cumulative_score": score + tag_dict[tag_id]["cumulative_score"], "scores_total": tag_dict[tag_id]["scores_total"] + 1}  
        return item_tag_dict


def item_tag_scores_to_CSV(tag_dict, item_tag_dict):
    df = pd.DataFrame(columns=['item_id', 'tag_scores'])
    for item in item_tag_dict:
        tag_scores = [0.0] * len(tag_dict)
        for tag in item_tag_dict[item]:
            scores = item_tag_dict[item][tag]["cumulative_score"]
            total = item_tag_dict[item][tag]["scores_total"]
            tag_scores[tag_dict[tag]] = (scores - total) / (total * 4) 
        df = df.append({'item_id': item, 'tag_scores': tag_scores}, ignore_index=True)    
    df.to_csv(f'item_tag_scores_{TAGS_SELECTED}_tags.csv', index=False)
    return df

def get_item_tag_training_labels(tag_dict, item_tag_dict, tag_application_dict):
    df = pd.DataFrame(columns=['item_id', 'tag_scores'])
    for item in item_tag_dict:
        tag_scores = [0] * len(tag_dict)
        if item in tag_application_dict:
            for tag in tag_application_dict[item]:
                tag_scores[tag_dict[tag]] = 1
        for tag in item_tag_dict[item]:
            scores = item_tag_dict[item][tag]["cumulative_score"]
            total = item_tag_dict[item][tag]["scores_total"]
            tag_scores[tag_dict[tag]] = int((scores - total) / (total * 4) >= 0.5) 
        df = df.append({'item_id': item, 'tag_scores': tag_scores}, ignore_index=True)
        df.to_csv(f'item_tag_train_{TAGS_SELECTED}_tags.csv', index=False)    
    return df

selected_tags, selected_items = select_tags_and_items()
item_tag_dict = get_item_tag_scores(selected_tags, selected_items)
tag_application_dict = get_tag_applications_per_item(selected_tags, selected_items)
review_df, item_ids = get_reviews_for_tenfolds(item_tag_dict)

tag_dict = get_tag_indexes(selected_tags)
item_tag_scores_to_CSV(tag_dict, item_tag_dict)
score_df = get_item_tag_training_labels(tag_dict, item_tag_dict, tag_application_dict)
data = pd.merge(review_df, score_df, left_on='item_id', right_on='item_id', how='left')
data.to_csv(f'sampled_reviews_with_scores_{TAGS_SELECTED}.csv', index=False)

json_ids = json.dumps(item_ids)

with open(f"tenfold_ids.json_{TAGS_SELECTED}", "w") as file:
    file.write(json_ids)

