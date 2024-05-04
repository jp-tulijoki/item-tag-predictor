import json
import pandas as pd

metadata_path = ""
output_file = ""

with open(metadata_path, "r") as file:
    titles = []
    for line in file:
        json_dict = json.loads(line)
        title_without_year = json_dict["title"].split("(")[0]
        titles.append({"item_id": json_dict["item_id"], "title": title_without_year})

df = pd.DataFrame(titles)

df.to_csv(output_file, index=False)