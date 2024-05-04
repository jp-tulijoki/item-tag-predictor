import pandas as pd
from json import loads
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

TENFOLD = 0

REVIEW_EMBEDDINGS_FILE = ""
TAG_EMBEDDINGS_FILE = ""

train_data = pd.read_csv(f"train{TENFOLD}.csv").rename(columns={"movieId": "item_id"})#.groupby(["tag", "item_id"]).mean().reset_index() # uncomment for better performance, will be handled when the files are combined later  
test_data = pd.read_csv(f"test{TENFOLD}.csv").rename(columns={"movieId": "item_id"})
review_embeddings = pd.read_csv(REVIEW_EMBEDDINGS_FILE)
tag_embeddings = pd.read_csv(TAG_EMBEDDINGS_FILE)

train_data = pd.merge(train_data, tag_embeddings, how="left", on="tag")
test_data = pd.merge(test_data, tag_embeddings, how="left", on="tag")

merged_train = pd.merge(train_data, review_embeddings, how="left", on="item_id")
merged_test = pd.merge(test_data, review_embeddings, how="left", on="item_id")

def get_cosine_reviews(row, data, merged):
    tag = data["tag"][row]
    print(tag)
    item_id = data["item_id"][row]
    embedding = np.vstack(merged[(merged["item_id"] == item_id) & (merged["tag"] == tag)]["embedding"].apply(lambda r: np.array(loads(r))))
    try:
        relevant_embeddings = np.vstack(merged[(merged["tag"] == tag) & (merged["tag_exists"] > 0)]["embedding"].apply(lambda r: np.array(loads(r))))
        cosine_sim = cosine_similarity(relevant_embeddings, embedding)
        cosine_sim = cosine_sim[:, 0].mean()
        return cosine_sim
    except:    
        return 0.0
    
def get_cosine_tag(row, data):
    item_id = data["item_id"][row]
    print(item_id)
    tag_embedding = np.array(loads(data["tag_embedding"][row])).reshape(1, -1)
    try:
        relevant_embeddings = np.vstack(data[(data["item_id"] == item_id) & (data["tag_exists"] > 0)]["tag_embedding"].apply(lambda r: np.array(loads(r))))
        cosine_sim = cosine_similarity(relevant_embeddings, tag_embedding)
        return cosine_sim[:, 0].mean()
    except:    
        return 0.0    

test_data["cosine_reviews"] = test_data.index.map(lambda i: get_cosine_reviews(i, test_data, merged_test))
train_data["cosine_reviews"] = train_data.index.map(lambda i: get_cosine_reviews(i, train_data, merged_train))
test_data["cosine_tag"] = test_data.index.map(lambda i: get_cosine_tag(i, test_data))
train_data["cosine_tag"] = train_data.index.map(lambda i: get_cosine_tag(i, train_data))

train_data = train_data.drop(columns=["tag_embedding"])
test_data = test_data.drop(columns=["tag_embedding"])

train_data.to_csv(f"train{TENFOLD}_modified.csv", index=False)
test_data.to_csv(f"test{TENFOLD}_modified.csv", index=False)
