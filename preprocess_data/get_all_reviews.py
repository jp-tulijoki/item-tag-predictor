import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

REVIEW_PATH = "" 
OUTPUT_PREFIX = ""  

def get_reviews():
    with open(REVIEW_PATH) as file:
        reviews = []
        for line in file:
            json_dict = json.loads(line)
            json_dict["txt"] = json_dict["txt"].replace('\r', '').replace('\t', ' ')
            reviews.append(json_dict)
        df = pd.DataFrame.from_records(reviews).drop_duplicates(subset=["txt"]).sample(frac=1, random_state=42)
        step_size = int(df.shape[0] / 10)
        for i in range(10):
            df[i*step_size: (i+1)*step_size].to_csv(f"{OUTPUT_PREFIX}{i}.csv", index=False)

get_reviews()