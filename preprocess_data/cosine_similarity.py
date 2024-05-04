import pandas as pd
from json import loads
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

TAG_EMBEDDINGS = ""
OTHER_EMBEDDINGS = "" # title or review embeddings
OUTPUT_FILE = ""

tags = pd.read_csv(TAG_EMBEDDINGS)
emb = pd.read_csv(OTHER_EMBEDDINGS)

tags["embedding"] = tags["tag_embedding"].map(loads).map(lambda r: np.array(r))
emb["embedding"] = emb["embedding"].map(loads).map(lambda r: np.array(r))

tag_names = tags["tag"].to_list()

emb = emb.sort_values(by="item_id")

result = pd.DataFrame(columns=["tag", "item_id", "cosine"])

for tag_name in tag_names:
    print(tag_name)
    tmp = emb.copy()
    tag_embedding = tags[tags["tag"] == tag_name]["embedding"].values[0].reshape(1, -1)
    cosine_sim = cosine_similarity(tmp["embedding"].tolist(), tag_embedding)
    tmp["cosine"] = cosine_sim[:, 0]
    tmp = tmp.drop(columns="embedding").groupby("item_id").mean().reset_index()
    tmp["tag"] = tag_name
    result = pd.concat([result, tmp])

result.to_csv(OUTPUT_FILE, index=False)