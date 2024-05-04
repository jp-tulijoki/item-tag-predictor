import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

review_path = ""
tenfolds_path = ""
output_file = ""

def get_reviews(items):
    with open(review_path) as file:
        reviews = []
        reviews_per_item = {}
        for line in file:
            json_dict = json.loads(line)
            id = json_dict["item_id"]
            if id not in items:
                continue
            if id not in reviews_per_item:
                reviews_per_item[id] = 1
            if reviews_per_item[id] < 10:
                json_dict["txt"] = json_dict["txt"].replace('\r', '').replace('\t', ' ')
                reviews.append(json_dict)
                reviews_per_item[id] = reviews_per_item[id] + 1
        df = pd.DataFrame.from_records(reviews)
        df.to_csv(output_file, index=False)

train = pd.read_csv(f"{tenfolds_path}/train0.csv")
test = pd.read_csv(f"{tenfolds_path}/test0.csv").rename(columns={"movieId": "item_id"})
items = pd.concat([train, test])["item_id"].unique().astype("int32").tolist()

get_reviews(items)
