from sentence_transformers import SentenceTransformer
import pandas as pd
from torch import cuda

TITLE_FILE = ""
OUTPUT_FILE = ""

data = pd.read_csv(TITLE_FILE)

titles = data["title"].to_list()
item_ids = data["item_id"].to_list()

model = SentenceTransformer("msmarco-distilbert-base-v4")

embeddings = model.encode(titles, show_progress_bar=True, device="cuda" if cuda.is_available() else "cpu")

embeddings_df = pd.DataFrame({"item_id": item_ids, "embedding": embeddings.tolist()})
embeddings_df.to_csv(OUTPUT_FILE)