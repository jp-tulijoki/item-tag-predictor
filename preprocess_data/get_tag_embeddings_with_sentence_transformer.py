from sentence_transformers import SentenceTransformer
import pandas as pd
import json

tag_path = ""
output_file = ""

def get_tags():
    with open(tag_path) as file:
        tags = []
        for line in file:
            json_dict = json.loads(line)
            tags.append(json_dict['tag'])  
        return tags 

tags = get_tags()

model = SentenceTransformer('msmarco-distilbert-base-v4')

embeddings = model.encode(tags, show_progress_bar=True)

embeddings_df = pd.DataFrame({"tag": tags, "embedding": embeddings.tolist()})
embeddings_df.to_csv(output_file, index=False)