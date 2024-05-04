import pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_error

EMBEDDINGS_MODEL = ""
TEST_FILE_PATH = ""
FEATURES = ["cosine_tag_review_mean", "cosine_tag_review_max", "cosine_title", "cosine_reviews", "cosine_tag"]
COMBINATIONS = list(product([True, False], repeat=len(FEATURES)))
TAGDL_MEAN = 0.846347694 # for movies
# TAGDL_MEAN = 0.751235945 # for books, uncomment to use

def select_features():
    selected = []
    for feature in feature_set.keys():
        if feature_set[feature]:
            selected.append(feature)
    return selected 

results = []
for i, combination in enumerate(COMBINATIONS):

    print(f"combination: {combination}")
    feature_set = dict(zip(FEATURES, combination))
    selected_features = select_features()
    prediction_folder = "_".join(["tag_dl"] + selected_features)
    maes = []

    for TENFOLD in range(10):
        test = pd.read_csv(f"{TEST_FILE_PATH}/test{TENFOLD}.csv").rename(columns={"movieId" : "item_id"})
        test = test[["tag", "targets", "item_id"]] 
        predictions = pd.read_csv(f"predictions/{prediction_folder}_{EMBEDDINGS_MODEL}/predictions{TENFOLD}.csv")
        predictions.predictions = predictions.predictions * 4 + 1
        evaluation = pd.merge(test, predictions, on=["tag", "item_id"], how="left")
        evaluation["error"] = evaluation.targets - evaluation.predictions
        mae = mean_absolute_error(evaluation.predictions, evaluation.targets)
        assert mae == evaluation['error'].abs().sum() / len(evaluation)
        print(mae)
        maes.append(mae) 
    overall_mae = sum(maes)/len(maes)
    print(overall_mae)
    tagdl_diff = (TAGDL_MEAN - overall_mae) / TAGDL_MEAN

    results.append({**feature_set, "overall_mae": overall_mae, "tagdl_diff": tagdl_diff})
    
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="overall_mae")

result_df.to_csv(f"result_summary_{EMBEDDINGS_MODEL}.csv", index=False)