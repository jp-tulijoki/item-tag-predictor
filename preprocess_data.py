import json
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

review_path = 'movie_dataset_public_final/raw/reviews.json'
tag_path = 'movie_dataset_public_final/raw/tags.json'

def get_reviews(items):
    with open(review_path) as file:
        reviews = []
        for line in file:
            json_dict = json.loads(line)
            id = json_dict["item_id"]
            if id not in items:
                continue
            json_dict["txt"] = json_dict["txt"].replace('\r', '').replace('\t', ' ')
            reviews.append(json_dict)
        return pd.DataFrame.from_records(reviews).drop_duplicates(subset=["txt"])

def get_data(train_scores, test_scores, reviews, tenfold):
    train_reviews = pd.DataFrame(columns=["item_id", "txt"])
    test_reviews = pd.DataFrame(columns=["item_id", "txt"])
    items = pd.concat([train_scores, test_scores])["item_id"].unique().tolist()
    test_items = test_scores["item_id"].unique().tolist()
    for item_id in items:
        reviews_for_item = reviews[reviews["item_id"] == item_id]
        fraction = 50 / reviews_for_item.shape[0] if reviews_for_item.shape[0] > 50 else 1
        sampled = reviews_for_item.sample(frac = fraction)
        sampled_test = sampled.sample(frac = 0.1)
        if sampled_test.shape[0] == 0 and item_id in test_items:
            sampled_test = sampled.sample(frac = 1 / sampled.shape[0])
        sampled_train = sampled.drop(sampled_test.index)
        train_reviews = pd.concat([train_reviews, sampled_train])
        test_reviews = pd.concat([test_reviews, sampled_test])
    train_data = pd.merge(train_reviews, train_scores, left_on='item_id', right_on='item_id', how='left')
    test_data = pd.merge(test_reviews, test_scores, left_on='item_id', right_on='item_id', how='left')
    assert test_items.sort() == test_data["item_id"].unique().tolist().sort() # assert test data has at least one review per item 
    assert pd.merge(train_reviews, test_reviews, on="txt", how="inner").empty # assert train and test reviews are separate
    train_data["targets"] = (train_data["targets"] - 1) / 4 # convert item-tag scores to scale [0,1]
    test_data["targets"] = (test_data["targets"] - 1) / 4 
    train_data.to_csv(f"reviews/{tenfold}/train_data_tag_id_{tag_id}.csv", index=False)
    test_data.to_csv(f"reviews/{tenfold}/test_data_tag_id_{tag_id}.csv", index=False)  

def get_tags():
    with open(tag_path) as file:
        tag_dict = {}
        tag_index = []
        for line in file:
            json_dict = json.loads(line)
            tag_dict[json_dict["tag"]] = json_dict["id"]
            tag_index.append(json_dict["id"])      
        return tag_dict, tag_index                  

def get_item_tag_scores(tag, tenfold):
    train_scores = train[train["tag"] == tag][["tag", "item_id", "targets"]].groupby(["tag", "item_id"]).mean().reset_index()
    test_scores = test[test["tag"] == tag][["tag", "item_id", "targets"]].groupby(["tag", "item_id"]).mean().reset_index()
    return train_scores, test_scores

tags, tag_index = get_tags()

for tenfold in range(10):
    train = pd.read_csv(f"movie_dataset_public_final/processed/10folds/train{tenfold}.csv")
    test = pd.read_csv(f"movie_dataset_public_final/processed/10folds/test{tenfold}.csv").rename(columns={"movieId": "item_id"})
    test_tags = test["tag"].unique().tolist()
    items = pd.concat([train, test])["item_id"].unique().astype("int32").tolist()
    reviews = get_reviews(items)

    for tag in test_tags:
        tag_id = tags[tag]
        print(f"processing tenfold {tenfold} tag {tag}")
        train_scores, test_scores = get_item_tag_scores(tag, tenfold)
        get_data(test_scores, test_scores, reviews, tenfold)

    review_list = sorted(list(filter(lambda r: "test" in r, os.listdir(f"reviews/{tenfold}"))))

    score_list = sorted(os.listdir(f"scores/{tenfold}"))

    with open(f"review_list{tenfold}.txt", "w") as file:
        for review in review_list:
            file.write(f"{review}\n")
            
    with open(f"score_list{tenfold}.txt", "w") as file:
        for score in score_list:
            file.write(f"{score}\n")         