import pandas as pd

MODEL = ""
PATH_TO_ORIGINAL_FILES = ""

COSINE_TAG_REVIEW_MEAN_FILE = ""
COSINE_TAG_REVIEW_MAX_FILE = ""
COSINE_TITLE_FILE = ""

cosine_tag_review_mean = pd.read_csv(COSINE_TAG_REVIEW_MEAN_FILE).rename(columns={"cosine" : "cosine_tag_review_mean"})
cosine_tag_review_max = pd.read_csv(COSINE_TAG_REVIEW_MAX_FILE).rename(columns={"cosine" : "cosine_tag_review_max"})
cosine_title = pd.read_csv(COSINE_TITLE_FILE).rename(columns={"cosine" : "cosine_title"})

for TENFOLD in range(10):
    print(TENFOLD)
    train_file = pd.read_csv(f"{PATH_TO_ORIGINAL_FILES}/train{TENFOLD}.csv")
    print(f"Original train shape: {train_file.shape}")
    test_file = pd.read_csv(f"{PATH_TO_ORIGINAL_FILES}/test{TENFOLD}.csv")
    print(f"Original test shape: {test_file.shape}")
    train_added = pd.read_csv(f"train{TENFOLD}_modified_{MODEL}.csv")[["tag", "item_id", "cosine_tag", "cosine_reviews"]].drop_duplicates(subset=["tag", "item_id"])
    test_added = pd.read_csv(f"test{TENFOLD}_modified_{MODEL}.csv")[["tag", "item_id", "cosine_tag", "cosine_reviews"]].drop_duplicates(subset=["tag", "item_id"])
    train_file = pd.merge(train_file, train_added, how="left", on=["tag", "item_id"])
    train_file = pd.merge(train_file, cosine_tag_review_mean, how="left", on=["tag","item_id"])
    train_file = pd.merge(train_file, cosine_tag_review_max, how="left", on=["tag","item_id"])
    train_file = pd.merge(train_file, cosine_title, how="left", on=["tag","item_id"])
    test_file = pd.merge(test_file, test_added, how="left", on=["tag","item_id"])
    test_file = pd.merge(test_file, cosine_tag_review_mean, how="left", on=["tag","item_id"])
    test_file = pd.merge(test_file, cosine_tag_review_max, how="left", on=["tag","item_id"])
    test_file = pd.merge(test_file, cosine_title, how="left", on=["tag","item_id"])
    print(f"Combined train shape: {train_file.shape}") # these should be similar to original shapes
    print(f"Combined test shape: {test_file.shape}")
    train_file.to_csv(f"train{TENFOLD}_combined_{MODEL}.csv", index=False)
    test_file.to_csv(f"test{TENFOLD}_combined_{MODEL}.csv", index=False)