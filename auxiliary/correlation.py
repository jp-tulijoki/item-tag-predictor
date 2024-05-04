import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# the combined train and test files including new features, tenfold 0 is enough
TRAIN_FILE = ""
TEST_FILE = ""

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

all = pd.concat((train, test))[["log_IMDB", "log_IMDB_nostem", "rating_similarity", "avg_rating", "tag_exists", "lsi_tags_75", "lsi_imdb_175", "tag_prob", "cosine_tag_review_mean", "cosine_tag_review_max", "cosine_title", "cosine_reviews", "cosine_tag", "targets"]]
all.rename(columns={"cosine_tag_review_mean": "mean (cos(t, R))", "cosine_tag_review_max": "max (cos(t, R))", "cosine_title": "cos(t, n)", "cosine_reviews": "mean (cos(R, R))", "cosine_tag": "mean (cos(t, T))"}, inplace=True)
correlation = all.corr(method="pearson")

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson correlation: books dataset, pre-trained pretrained msmarco-distilbert-base-v4")
plt.show()
